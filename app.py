
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback

app = Flask(__name__, static_folder='static')
CORS(app)  

print("Cargando modelos...")

try:
    # Cargar modelo entrenado
    model = joblib.load('modelo_final_sabor_chapaco.pkl')
    print("✓ Modelo cargado")
    
    # Cargar scaler
    scaler = joblib.load('scaler_sabor_chapaco.pkl')
    print("✓ Scaler cargado")
    
    # Cargar encoders
    label_encoders = joblib.load('label_encoders_sabor_chapaco.pkl')
    print("✓ Label encoders cargados")
    
    # Cargar información del modelo
    model_info = joblib.load('model_info.pkl')
    print("✓ Model info cargado")
    
    # Cargar polynomial features si es necesario
    if model_info['is_polynomial']:
        poly_features = joblib.load('polynomial_features.pkl')
        print(f"✓ Polynomial features cargado (grado {model_info['polynomial_degree']})")
    else:
        poly_features = None
    
    print("\n✓✓✓ Todos los modelos cargados exitosamente ✓✓✓\n")
    
except FileNotFoundError as e:
    print(f"✗ ERROR: No se encontró el archivo {e.filename}")
    print("Asegúrate de que todos los archivos .pkl estén en el directorio actual")
    exit(1)
except Exception as e:
    err_msg = str(e)
    print(f"✗ ERROR al cargar modelos: {err_msg}")
    # Mensaje específico cuando falta scikit-learn (módulo 'sklearn') durante unpickle
    try:
        missing_module = isinstance(e, ModuleNotFoundError) and (getattr(e, 'name', '') == 'sklearn' or 'sklearn' in err_msg)
    except Exception:
        missing_module = False

    if missing_module:
        print("\nParece que falta la librería 'scikit-learn' (módulo 'sklearn').")
        print("Por favor activa el entorno virtual del proyecto y instala las dependencias:")
        print("  En PowerShell (desde la carpeta del proyecto):")
        print("    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\\.venv\\Scripts\\Activate.ps1")
        print("    pip install -r requerimientos.txt")
        print("  O alternativamente en cmd:\\")
        print("    .\\.venv\\Scripts\\activate && pip install -r requerimientos.txt")

    traceback.print_exc()
    exit(1)


# ==============================
# FEATURE ENGINEERING
# ==============================

def engineer_features(data_dict):
    
    df = pd.DataFrame([data_dict])
    
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['mes'] = df['fecha'].dt.month
    df['dia'] = df['fecha'].dt.day
    
    df['hora_num'] = df['hora_servicio'].astype(int)
    df['minuto'] = 0  
    
    df['es_almuerzo'] = ((df['hora_num'] >= 11) & (df['hora_num'] <= 15)).astype(int)
    df['es_cena'] = ((df['hora_num'] >= 18) & (df['hora_num'] <= 22)).astype(int)
    df['es_desayuno'] = ((df['hora_num'] >= 6) & (df['hora_num'] <= 10)).astype(int)
    
    df['trimestre'] = df['mes'].apply(lambda x: (x-1)//3 + 1)
    df['es_fin_mes'] = (df['fecha'].dt.day > 25).astype(int)
    
    df['temporada_turistica'] = df['mes'].apply(lambda x: 1 if x in [5,6,7,8,9,10] else 0)
    
    # Apartado para codificación cíclica)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['hora_sin'] = np.sin(2 * np.pi * df['hora_num'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora_num'] / 24)
    
    dias_map = {
        'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
        'Viernes': 4, 'Sábado': 5, 'Domingo': 6
    }
    df['dia_semana_num'] = df['dia_semana'].map(dias_map)
    df['dia_semana_num'] = df['dia_semana_num'].fillna(-1).astype(int)
    
    return df


def encode_and_prepare_features(df):

    categorical_cols = ['plato', 'condicion_climatica', 'evento_local', 'tipo_promocion']
    
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            try:
                df[f'{col}_encoded'] = le.transform(df[col].astype(str))
            except ValueError:
       
                print(f"⚠️ Categoría no vista en {col}: {df[col].values[0]}, usando categoría por defecto")
                df[f'{col}_encoded'] = 0
    
    numeric_features = [
        'hora_num', 'minuto', 'es_fin_semana',
        'mes', 'dia_semana_num',
        'es_almuerzo', 'es_cena', 'es_desayuno',
        'trimestre', 'temporada_turistica', 'es_fin_mes',
        'mes_sin', 'mes_cos', 'hora_sin', 'hora_cos'
    ]
    
    categorical_encoded = [
        'plato_encoded', 'condicion_climatica_encoded',
        'evento_local_encoded', 'tipo_promocion_encoded'
    ]
    
    feature_cols = numeric_features + categorical_encoded
    
    X = df[feature_cols]
    
    # Normalización y transformación polinómica
    X_scaled = scaler.transform(X)
    
    if poly_features is not None:
        X_scaled = poly_features.transform(X_scaled)
    
    return X_scaled

# RUTAS DE LA API

@app.route('/')
def home():
    """Página principal - sirve el HTML"""
    return send_from_directory('.', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    
    try:

        data = request.get_json()
        
        print(f"\n📥 Nueva solicitud de predicción recibida:")
        print(f"   Plato: {data.get('plato')}")
        print(f"   Fecha: {data.get('fecha')}")
        print(f"   Hora: {data.get('hora')}")
        
        required_fields = ['plato', 'fecha', 'hora', 'dia_semana', 'clima', 
                          'evento', 'promocion', 'fin_semana']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Faltan campos requeridos: {", ".join(missing_fields)}'
            }), 400
        
        # Preparar datos para el modelo
        input_data = {
            'fecha': data['fecha'],
            'plato': data['plato'],
            'hora_servicio': int(data['hora']),
            'dia_semana': data['dia_semana'],
            'condicion_climatica': data['clima'],
            'evento_local': data['evento'],
            'tipo_promocion': data['promocion'],
            'es_fin_semana': int(data['fin_semana'])
        }
        
        df_features = engineer_features(input_data)
        X_processed = encode_and_prepare_features(df_features)
        prediction = model.predict(X_processed)[0]
        
        prediction = max(0, prediction)
        
        # Calcular recomendaciones de stock
        # Margen de seguridad del 15%
        margen = max(2, int(prediction * 0.15))
        stock_min = max(0, int(prediction - margen))
        stock_max = int(prediction + margen)
        
        r2_score = model_info['test_metrics']['R2']
        confianza = int(r2_score * 100)
        
        # Preparar respuesta
        response = {
            'success': True,
            'prediction': round(prediction, 1),
            'prediction_rounded': int(round(prediction)),
            'stock_min': stock_min,
            'stock_max': stock_max,
            'margen_seguridad': margen,
            'confianza': confianza,
            'model_name': model_info['model_name'],
            'model_r2': round(r2_score, 4),
            'model_mae': round(model_info['test_metrics']['MAE'], 2),
            'fecha_prediccion': data['fecha'],
            'plato': data['plato']
        }
        
        print(f"✅ Predicción exitosa: {response['prediction_rounded']} porciones")
        print(f"   Rango recomendado: {stock_min} - {stock_max} porciones")
        print(f"   Confianza: {confianza}%\n")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ ERROR en predicción: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error al procesar la predicción. Por favor verifica los datos.'
        }), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Retorna información sobre el modelo cargado
    """
    
    try:
        info = {
            'success': True,
            'model_name': model_info['model_name'],
            'is_polynomial': model_info['is_polynomial'],
            'polynomial_degree': model_info.get('polynomial_degree'),
            'metrics': {
                'R2': round(model_info['test_metrics']['R2'], 4),
                'MAE': round(model_info['test_metrics']['MAE'], 2),
                'RMSE': round(model_info['test_metrics']['RMSE'], 2),
                'MAPE': round(model_info['test_metrics']['MAPE'], 2)
            },
            'best_params': model_info.get('best_params', {}),
            'feature_count': len(model_info['feature_names']),
            'trained_date': model_info.get('trained_date', 'N/A')
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Endpoint de health check para verificar estado del servidor
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'encoders_loaded': label_encoders is not None,
        'version': '2.0.0',
        'model_type': model_info.get('model_name', 'Unknown')
    })


@app.route('/api/platos', methods=['GET'])
def get_platos():
    """
    Retorna lista de platos disponibles
    """
    platos = [
        'Ranga Ranga',
        'Sajta de Pollo',
        'Chancho a la Cruz',
        'Picante de Lengua',
        'Chicharrón de Cerdo'
    ]
    
    return jsonify({
        'success': True,
        'platos': platos
    })


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Retorna estadísticas generales del modelo
    """
    try:
        stats = {
            'success': True,
            'total_predictions': 0,  # Podrías implementar un contador
            'model_accuracy': round(model_info['test_metrics']['R2'] * 100, 2),
            'average_error': round(model_info['test_metrics']['MAE'], 2),
            'model_version': model_info.get('model_name', 'Unknown'),
            'last_update': model_info.get('trained_date', 'N/A')
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# MANEJO DE ERRORES


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint no encontrado'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor'
    }), 500

# EJECUCIÓN DEL SERVIDOR

if __name__ == '__main__':
    print("SERVIDOR FLASK - RESTAURANTE SABOR CHAPACO")
    
    print(f"🤖 Modelo cargado: {model_info['model_name']}")
    print(f"📊 R² Score: {model_info['test_metrics']['R2']:.4f}")
    print(f"📏 MAE: {model_info['test_metrics']['MAE']:.2f} porciones")
    print(f"📐 RMSE: {model_info['test_metrics']['RMSE']:.2f}")
    print(f"📈 MAPE: {model_info['test_metrics']['MAPE']:.2f}%")
    
    print(f"\n🌐 Endpoints disponibles:")
    print(f"   - POST /api/predict        → Realizar predicción")
    print(f"   - GET  /api/model-info     → Información del modelo")
    print(f"   - GET  /api/health         → Estado del servidor")
    print(f"   - GET  /api/platos         → Lista de platos")
    print(f"   - GET  /api/statistics     → Estadísticas del sistema")
    
    print(f"\n🚀 Iniciando servidor Flask...")
    print(f"📍 URL: http://localhost:5000")
    print(f"📍 API: http://localhost:5000/api/")
    print(f"\n💡 Presiona CTRL+C para detener el servidor\n")
    
    app.run(
        host='0.0.0.0',  
        port=5000,
        debug=True  
    )