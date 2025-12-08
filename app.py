"""
Backend Flask - Sabor Chapaco (VERSIÓN CORREGIDA)
Compatible con modelos generados por fix_modelos.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import traceback

import glob
import json
from datetime import datetime
from flask import jsonify


app = Flask(__name__, static_folder='static')
CORS(app)

print("="*80)
print("🚀 CARGANDO SISTEMA - SABOR CHAPACO")
print("="*80)

# ============================================================================
# CARGA DE MODELOS
# ============================================================================

try:
    # Cargar modelo principal (Pipeline completo)
    model = joblib.load('modelo_final_sabor_chapaco.pkl')
    print("✅ Modelo Polinomial cargado")
    
    # Cargar componentes auxiliares
    scaler = joblib.load('scaler_sabor_chapaco.pkl')
    print("✅ Scaler cargado")
    
    label_encoders = joblib.load('label_encoders_sabor_chapaco.pkl')
    print("✅ Label encoders cargados")
    
    model_info = joblib.load('model_info.pkl')
    print("✅ Model info cargado")
    
    # Intentar cargar polynomial features (opcional)
    try:
        poly_features = joblib.load('polynomial_features.pkl')
        print("✅ Polynomial features cargado")
    except:
        poly_features = None
        print("⚠️  Polynomial features no encontrado (el modelo funciona igual)")
    
    print("\n✓✓✓ SISTEMA LISTO ✓✓✓\n")
    
except Exception as e:
    print(f"✗ ERROR al cargar modelos: {e}")
    print("\n⚠️  Ejecuta primero: python fix_modelos.py")
    exit(1)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(data_dict):
    """Crea features exactamente igual que en entrenamiento"""
    df = pd.DataFrame([data_dict])
    
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['mes'] = df['fecha'].dt.month
    df['dia'] = df['fecha'].dt.day
    
    df['hora_num'] = df['hora_servicio'].astype(int)
    df['minuto'] = 0
    
    # Franjas horarias
    df['es_almuerzo'] = ((df['hora_num'] >= 11) & (df['hora_num'] <= 15)).astype(int)
    df['es_cena'] = ((df['hora_num'] >= 18) & (df['hora_num'] <= 22)).astype(int)
    df['es_desayuno'] = ((df['hora_num'] >= 6) & (df['hora_num'] <= 10)).astype(int)
    
    # Features temporales
    df['trimestre'] = df['mes'].apply(lambda x: (x-1)//3 + 1)
    df['es_fin_mes'] = (df['fecha'].dt.day > 25).astype(int)
    df['temporada_turistica'] = df['mes'].apply(lambda x: 1 if x in [5,6,7,8,9,10] else 0)
    
    # Features cíclicas
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['hora_sin'] = np.sin(2 * np.pi * df['hora_num'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora_num'] / 24)
    
    # Día de la semana
    dias_map = {
        'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
        'Viernes': 4, 'Sábado': 5, 'Domingo': 6
    }
    df['dia_semana_num'] = df['dia_semana'].map(dias_map)
    df['dia_semana_num'] = df['dia_semana_num'].fillna(-1).astype(int)
    
    return df


def encode_and_prepare_features(df):
    """Codifica variables categóricas y prepara features"""
    
    categorical_cols = ['plato', 'condicion_climatica', 'evento_local', 'tipo_promocion']
    
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            try:
                df[f'{col}_encoded'] = le.transform(df[col].astype(str))
            except ValueError:
                print(f"⚠️ Categoría no vista en {col}: {df[col].values[0]}, usando 0")
                df[f'{col}_encoded'] = 0
    
    # Lista exacta de features (mismo orden que en entrenamiento)
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
    
    return X


# ============================================================================
# RUTAS DE LA API
# ============================================================================

@app.route('/')
def home():
    """Página principal"""
    return send_from_directory('.', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint principal de predicción"""
    
    try:
        data = request.get_json()
        
        print(f"\n📥 Predicción para: {data.get('plato')} - {data.get('fecha')}")
        
        # Validar campos requeridos
        required_fields = ['plato', 'fecha', 'hora', 'dia_semana', 'clima', 
                          'evento', 'promocion', 'fin_semana']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Faltan campos: {", ".join(missing_fields)}'
            }), 400
        
        # Preparar datos
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
        
        # Feature engineering
        df_features = engineer_features(input_data)
        X_processed = encode_and_prepare_features(df_features)
        
        # PREDICCIÓN con el Pipeline completo
        prediction = model.predict(X_processed)[0]
        prediction = max(0, prediction)
        
        # Calcular recomendaciones de stock
        margen = max(2, int(prediction * 0.15))
        stock_min = max(0, int(prediction - margen))
        stock_max = int(prediction + margen)
        
        # Nivel de confianza del modelo
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
            'plato': data['plato'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Predicción: {response['prediction_rounded']} porciones")
        print(f"   Rango: {stock_min} - {stock_max} porciones")
        print(f"   Confianza: {confianza}%\n")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error al procesar la predicción'
        }), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Predicción para múltiples días"""
    
    try:
        data = request.get_json()
        dias = int(data.get('dias', 7))
        plato = data['plato']
        fecha_inicio = datetime.strptime(data['fecha_inicio'], '%Y-%m-%d')
        
        predicciones_semana = []
        
        for i in range(dias):
            fecha = fecha_inicio + timedelta(days=i)
            
            input_dia = {
                'fecha': fecha.strftime('%Y-%m-%d'),
                'plato': plato,
                'hora_servicio': 19,
                'dia_semana': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 
                              'Viernes', 'Sábado', 'Domingo'][fecha.weekday()],
                'condicion_climatica': 'Soleado',
                'evento_local': 'Ninguno',
                'tipo_promocion': 'Ninguna',
                'es_fin_semana': 1 if fecha.weekday() >= 4 else 0
            }
            
            df_features = engineer_features(input_dia)
            X = encode_and_prepare_features(df_features)
            pred = model.predict(X)[0]
            
            predicciones_semana.append({
                'fecha': input_dia['fecha'],
                'dia_semana': input_dia['dia_semana'],
                'prediccion': round(max(0, pred), 1)
            })
        
        return jsonify({
            'success': True,
            'plato': plato,
            'periodo': f"{dias} días",
            'predicciones': predicciones_semana,
            'total_estimado': sum([p['prediccion'] for p in predicciones_semana]),
            'promedio_diario': round(np.mean([p['prediccion'] for p in predicciones_semana]), 1)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Información del modelo"""
    
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
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Estado del servidor"""
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
    """Lista de platos disponibles"""
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
    """Estadísticas del sistema"""
    try:
        stats = {
            'success': True,
            'total_predictions': 0,
            'model_accuracy': round(model_info['test_metrics']['R2'] * 100, 2),
            'average_error': round(model_info['test_metrics']['MAE'], 2),
            'model_version': model_info.get('model_name', 'Unknown'),
            'last_update': model_info.get('trained_date', 'N/A')
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ENDPOINTS DE ALERTAS
# ============================================================================

@app.route('/api/alertas/ultimas', methods=['GET'])
def obtener_ultimas_alertas():
    """
    Retorna las alertas más recientes generadas
    GET /api/alertas/ultimas
    """
    try:
        # Buscar archivo de alertas más reciente
        archivos_alertas = glob.glob('alertas_*.json')
        
        if not archivos_alertas:
            return jsonify({
                'success': True,
                'message': 'No hay alertas generadas aún',
                'alertas': [],
                'total': 0
            })
        
        # Ordenar por fecha (más reciente primero)
        archivo_mas_reciente = max(archivos_alertas)
        
        with open(archivo_mas_reciente, 'r', encoding='utf-8') as f:
            datos_alertas = json.load(f)
        
        return jsonify({
            'success': True,
            'fecha_generacion': datos_alertas['fecha_generacion'],
            'total_alertas': datos_alertas['total_alertas'],
            'alertas_criticas': datos_alertas['alertas_criticas'],
            'alertas': datos_alertas['alertas']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/alertas/criticas', methods=['GET'])
def obtener_alertas_criticas():
    """
    Retorna solo las alertas CRÍTICAS y ALTAS
    GET /api/alertas/criticas
    """
    try:
        archivos_alertas = glob.glob('alertas_*.json')
        
        if not archivos_alertas:
            return jsonify({
                'success': True,
                'alertas': [],
                'total': 0
            })
        
        archivo_mas_reciente = max(archivos_alertas)
        
        with open(archivo_mas_reciente, 'r', encoding='utf-8') as f:
            datos_alertas = json.load(f)
        
        # Filtrar solo críticas y altas
        alertas_criticas = [
            a for a in datos_alertas['alertas']
            if a['nivel'] in ['CRITICO', 'ALTO']
        ]
        
        return jsonify({
            'success': True,
            'fecha_generacion': datos_alertas['fecha_generacion'],
            'total_criticas': len(alertas_criticas),
            'alertas': alertas_criticas
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/alertas/por-plato/<plato>', methods=['GET'])
def obtener_alertas_por_plato(plato):
    """
    Retorna alertas para un plato específico
    GET /api/alertas/por-plato/Ranga%20Ranga
    """
    try:
        archivos_alertas = glob.glob('alertas_*.json')
        
        if not archivos_alertas:
            return jsonify({
                'success': True,
                'plato': plato,
                'alertas': [],
                'total': 0
            })
        
        archivo_mas_reciente = max(archivos_alertas)
        
        with open(archivo_mas_reciente, 'r', encoding='utf-8') as f:
            datos_alertas = json.load(f)
        
        # Filtrar por plato
        alertas_plato = [
            a for a in datos_alertas['alertas']
            if a.get('plato') == plato
        ]
        
        return jsonify({
            'success': True,
            'plato': plato,
            'total': len(alertas_plato),
            'alertas': alertas_plato
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/alertas/resumen', methods=['GET'])
def obtener_resumen_alertas():
    """
    Retorna resumen estadístico de alertas
    GET /api/alertas/resumen
    """
    try:
        archivos_alertas = glob.glob('alertas_*.json')
        
        if not archivos_alertas:
            return jsonify({
                'success': True,
                'mensaje': 'No hay alertas generadas',
                'resumen': {
                    'total': 0,
                    'criticas': 0,
                    'por_tipo': {},
                    'por_plato': {}
                }
            })
        
        archivo_mas_reciente = max(archivos_alertas)
        
        with open(archivo_mas_reciente, 'r', encoding='utf-8') as f:
            datos_alertas = json.load(f)
        
        alertas = datos_alertas['alertas']
        
        # Análisis estadístico
        por_nivel = {}
        por_tipo = {}
        por_plato = {}
        
        for alerta in alertas:
            # Por nivel
            nivel = alerta['nivel']
            por_nivel[nivel] = por_nivel.get(nivel, 0) + 1
            
            # Por tipo
            tipo = alerta['tipo']
            por_tipo[tipo] = por_tipo.get(tipo, 0) + 1
            
            # Por plato
            plato = alerta.get('plato', 'General')
            por_plato[plato] = por_plato.get(plato, 0) + 1
        
        return jsonify({
            'success': True,
            'fecha_generacion': datos_alertas['fecha_generacion'],
            'resumen': {
                'total': len(alertas),
                'criticas': datos_alertas['alertas_criticas'],
                'por_nivel': por_nivel,
                'por_tipo': por_tipo,
                'por_plato': por_plato
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/alertas/generar-ahora', methods=['POST'])
def generar_alertas_manual():
    """
    Endpoint para generar alertas manualmente (sin esperar scheduler)
    POST /api/alertas/generar-ahora
    """
    try:
        # Importar función de generación
        from sistema_alertas_automatico import ejecutar_alertas_ahora
        
        # Ejecutar en un thread separado para no bloquear
        import threading
        thread = threading.Thread(target=ejecutar_alertas_ahora)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Generación de alertas iniciada',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Manejo de errores
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint no encontrado'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Error interno del servidor'}), 500


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("🌐 SERVIDOR FLASK - RESTAURANTE SABOR CHAPACO")
    print("="*80)
    
    print(f"🤖 Modelo: {model_info['model_name']}")
    print(f"📊 R² Score: {model_info['test_metrics']['R2']:.4f}")
    print(f"📏 MAE: {model_info['test_metrics']['MAE']:.2f} porciones")
    print(f"📐 RMSE: {model_info['test_metrics']['RMSE']:.2f}")
    print(f"📈 MAPE: {model_info['test_metrics']['MAPE']:.2f}%")
    
    print(f"\n🌐 Endpoints disponibles:")
    print(f"   - POST /api/predict        → Predicción individual")
    print(f"   - POST /api/predict-batch  → Predicción múltiples días")
    print(f"   - GET  /api/model-info     → Info del modelo")
    print(f"   - GET  /api/health         → Estado del servidor")
    print(f"   - GET  /api/platos         → Lista de platos")
    print(f"   - GET  /api/statistics     → Estadísticas")
    
    print(f"\n🚀 Iniciando servidor...")
    print(f"📍 URL: http://localhost:5000")
    print(f"\n💡 Presiona CTRL+C para detener\n")
    print("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=True)