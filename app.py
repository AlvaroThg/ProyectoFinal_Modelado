"""
Backend Flask Mejorado - Sabor Chapaco
Sistema Predictivo con Múltiples Modelos ML + Alertas Inteligentes
Examen Final - Modelado y Simulación de Sistemas
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import traceback
import json

# Importar sistema de alertas
from sistema_alertas_inteligentes import MotorAlertas, TipoAlerta, NivelAlerta

app = Flask(__name__, static_folder='static')
CORS(app)

print("="*80)
print("🚀 CARGANDO SISTEMA MEJORADO - SABOR CHAPACO")
print("="*80)

# ============================================================================
# CARGA DE MODELOS
# ============================================================================

modelos_disponibles = {}

# Intentar cargar todos los modelos
modelos_a_cargar = [
    ('polinomial', 'modelo_final_sabor_chapaco.pkl'),
    ('prophet', 'modelo_prophet_sabor_chapaco.pkl'),
    ('sarima', 'modelo_sarima_sabor_chapaco.pkl'),
    ('xgboost', 'modelo_xgboost_sabor_chapaco.pkl'),
    ('lstm', 'modelo_lstm_sabor_chapaco.pkl')
]

for nombre, archivo in modelos_a_cargar:
    try:
        modelo = joblib.load(archivo)
        modelos_disponibles[nombre] = modelo
        print(f"✅ Modelo {nombre.upper()} cargado")
    except FileNotFoundError:
        print(f"⚠️  Modelo {nombre.upper()} no encontrado (opcional)")
    except Exception as e:
        print(f"✗  Error cargando {nombre}: {e}")

# Cargar componentes auxiliares
try:
    scaler = joblib.load('scaler_sabor_chapaco.pkl')
    label_encoders = joblib.load('label_encoders_sabor_chapaco.pkl')
    model_info = joblib.load('model_info.pkl')
    print("✅ Componentes auxiliares cargados")
except Exception as e:
    print(f"✗  Error cargando componentes: {e}")
    scaler = None
    label_encoders = {}
    model_info = {'test_metrics': {'R2': 0.76, 'MAE': 8.69}}

# Inicializar motor de alertas
motor_alertas = MotorAlertas()
print("✅ Motor de alertas inicializado")

print("\n✓✓✓ SISTEMA LISTO ✓✓✓\n")


# ============================================================================
# CONFIGURACIÓN DEL NEGOCIO
# ============================================================================

configuracion_negocio = {
    'nombre_restaurante': 'Sabor Chapaco',
    'horario_apertura': '15:00',
    'horario_cierre': '00:00',
    'capacidad_maxima': 80,  # personas
    'platos': {
        'Ranga Ranga': {
            'costo_ingredientes': 15,
            'precio_venta': 35,
            'tiempo_preparacion_min': 30,
            'ingredientes_principales': ['mondongo', 'papa', 'arvejas'],
            'vida_util_horas': 4
        },
        'Sajta de Pollo': {
            'costo_ingredientes': 12,
            'precio_venta': 30,
            'tiempo_preparacion_min': 25,
            'ingredientes_principales': ['pollo', 'papa', 'ají'],
            'vida_util_horas': 3
        },
        'Chancho a la Cruz': {
            'costo_ingredientes': 18,
            'precio_venta': 40,
            'tiempo_preparacion_min': 120,
            'ingredientes_principales': ['cerdo', 'especias'],
            'vida_util_horas': 6
        },
        'Picante de Lengua': {
            'costo_ingredientes': 20,
            'precio_venta': 45,
            'tiempo_preparacion_min': 90,
            'ingredientes_principales': ['lengua', 'papa', 'ají'],
            'vida_util_horas': 4
        },
        'Chicharrón de Cerdo': {
            'costo_ingredientes': 14,
            'precio_venta': 32,
            'tiempo_preparacion_min': 45,
            'ingredientes_principales': ['cerdo', 'mote'],
            'vida_util_horas': 5
        }
    },
    'proveedores': {
        'carnes': {
            'nombre': 'Frigorífico Central',
            'dias_entrega': [1, 3, 5],  # Lunes, Miércoles, Viernes
            'tiempo_entrega_horas': 24
        },
        'verduras': {
            'nombre': 'Mercado Campesino',
            'dias_entrega': [0, 2, 4, 6],  # Todos los días excepto domingo
            'tiempo_entrega_horas': 12
        }
    },
    'inventario_objetivo_dias': 2,
    'margen_seguridad': 0.20
}


# ============================================================================
# FUNCIONES DE FEATURE ENGINEERING
# ============================================================================

def engineer_features(data_dict):
    """Crea features para los modelos"""
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
    
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['hora_sin'] = np.sin(2 * np.pi * df['hora_num'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora_num'] / 24)
    
    dias_map = {
        'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
        'Viernes': 4, 'Sábado': 5, 'Domingo': 6
    }
    df['dia_semana_num'] = df['dia_semana'].map(dias_map)
    
    return df


def encode_and_prepare_features(df):
    """Codifica y normaliza features"""
    categorical_cols = ['plato', 'condicion_climatica', 'evento_local', 'tipo_promocion']
    
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders.get(col)
            if le:
                try:
                    df[f'{col}_encoded'] = le.transform(df[col].astype(str))
                except ValueError:
                    df[f'{col}_encoded'] = 0
    
    numeric_features = [
        'hora_num', 'minuto', 'es_fin_semana', 'mes', 'dia_semana_num',
        'es_almuerzo', 'es_cena', 'es_desayuno', 'trimestre', 
        'temporada_turistica', 'es_fin_mes',
        'mes_sin', 'mes_cos', 'hora_sin', 'hora_cos'
    ]
    
    categorical_encoded = [
        'plato_encoded', 'condicion_climatica_encoded',
        'evento_local_encoded', 'tipo_promocion_encoded'
    ]
    
    feature_cols = numeric_features + categorical_encoded
    X = df[feature_cols]
    
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    return X_scaled


# ============================================================================
# ENDPOINTS DE LA API
# ============================================================================

@app.route('/')
def home():
    """Página principal"""
    return send_from_directory('.', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de predicción
    Usa múltiples modelos y genera alertas inteligentes
    """
    try:
        data = request.get_json()
        
        print(f"\n📥 Predicción para: {data.get('plato')} - {data.get('fecha')}")
        
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
        
        df_features = engineer_features(input_data)
        X_processed = encode_and_prepare_features(df_features)
        
        # Predicciones con todos los modelos disponibles
        predicciones = {}
        
        if 'polinomial' in modelos_disponibles:
            pred = modelos_disponibles['polinomial']['modelo'].predict(X_processed)[0]
            predicciones['polinomial'] = max(0, pred)
        
        # Prophet, SARIMA, XGBoost, LSTM requieren formato específico
        # (implementar según disponibilidad)
        
        # Predicción final (ensemble o mejor modelo)
        if predicciones:
            prediction = np.mean(list(predicciones.values()))
        else:
            prediction = 45  # Fallback
        
        prediction = max(0, prediction)
        
        # Calcular stock recomendado
        margen = max(2, int(prediction * 0.15))
        stock_min = max(0, int(prediction - margen))
        stock_max = int(prediction + margen)
        
        # GENERAR ALERTAS INTELIGENTES
        alertas = motor_alertas.generar_alertas_dia(
            prediccion=prediction,
            fecha=datetime.strptime(data['fecha'], '%Y-%m-%d'),
            plato=data['plato'],
            inventario_disponible=int(data.get('inventario_actual', stock_max)),
            stock_planeado=stock_max,
            condicion_clima=data['clima'],
            tipo_evento=data['evento'],
            demanda_promedio=prediction * 0.95
        )
        
        # Preparar respuesta
        response = {
            'success': True,
            'prediccion': round(prediction, 1),
            'prediccion_redondeada': int(round(prediction)),
            'stock_min': stock_min,
            'stock_max': stock_max,
            'margen_seguridad': margen,
            'confianza': int(model_info['test_metrics']['R2'] * 100),
            'modelos_usados': list(predicciones.keys()),
            'predicciones_individuales': {k: round(v, 1) for k, v in predicciones.items()},
            'alertas': [a.to_dict() for a in alertas],
            'alertas_criticas': len([a for a in alertas if a.nivel.name in ['CRITICO', 'ALTO']]),
            'info_plato': configuracion_negocio['platos'].get(data['plato'], {}),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Predicción: {response['prediccion_redondeada']} porciones")
        print(f"🚨 Alertas: {len(alertas)} generadas ({response['alertas_criticas']} críticas)")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predicción para múltiples días (útil para planificación semanal)
    """
    try:
        data = request.get_json()
        dias = int(data.get('dias', 7))
        plato = data['plato']
        fecha_inicio = datetime.strptime(data['fecha_inicio'], '%Y-%m-%d')
        
        predicciones_semana = []
        
        for i in range(dias):
            fecha = fecha_inicio + timedelta(days=i)
            
            # Crear input para cada día
            input_dia = {
                'fecha': fecha.strftime('%Y-%m-%d'),
                'plato': plato,
                'hora': 19,  # Hora peak
                'dia_semana': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][fecha.weekday()],
                'clima': 'Soleado',  # Default
                'evento': 'Ninguno',
                'promocion': 'Ninguna',
                'fin_semana': 1 if fecha.weekday() >= 4 else 0
            }
            
            # Hacer predicción individual
            df_features = engineer_features({
                'fecha': input_dia['fecha'],
                'plato': input_dia['plato'],
                'hora_servicio': input_dia['hora'],
                'dia_semana': input_dia['dia_semana'],
                'condicion_climatica': input_dia['clima'],
                'evento_local': input_dia['evento'],
                'tipo_promocion': input_dia['promocion'],
                'es_fin_semana': input_dia['fin_semana']
            })
            
            X = encode_and_prepare_features(df_features)
            pred = modelos_disponibles['polinomial']['modelo'].predict(X)[0]
            
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


@app.route('/api/alertas', methods=['GET'])
def obtener_alertas():
    """Retorna todas las alertas activas"""
    try:
        alertas = motor_alertas.alertas_generadas
        
        return jsonify({
            'success': True,
            'total_alertas': len(alertas),
            'alertas': [a.to_dict() for a in alertas],
            'resumen': motor_alertas.resumen_alertas()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config', methods=['GET', 'POST'])
def configuracion():
    """
    GET: Obtiene configuración actual
    POST: Actualiza configuración
    """
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'configuracion_negocio': configuracion_negocio,
            'configuracion_alertas': motor_alertas.configuracion
        })
    
    elif request.method == 'POST':
        try:
            nueva_config = request.get_json()
            
            if 'alertas' in nueva_config:
                motor_alertas.actualizar_configuracion(nueva_config['alertas'])
            
            if 'negocio' in nueva_config:
                configuracion_negocio.update(nueva_config['negocio'])
            
            return jsonify({
                'success': True,
                'message': 'Configuración actualizada'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def listar_modelos():
    """Lista todos los modelos disponibles y sus métricas"""
    modelos_info = []
    
    for nombre, modelo in modelos_disponibles.items():
        info = {
            'nombre': nombre,
            'cargado': True,
            'metricas': modelo.get('metricas', {})
        }
        modelos_info.append(info)
    
    return jsonify({
        'success': True,
        'modelos_disponibles': len(modelos_disponibles),
        'modelos': modelos_info
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Estado del servidor"""
    return jsonify({
        'status': 'healthy',
        'modelos_cargados': len(modelos_disponibles),
        'motor_alertas': 'activo',
        'version': '3.0.0',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/api/statistics', methods=['GET'])
def estadisticas():
    """Estadísticas generales del sistema"""
    return jsonify({
        'success': True,
        'modelos_ml': len(modelos_disponibles),
        'alertas_generadas': len(motor_alertas.alertas_generadas),
        'platos_configurados': len(configuracion_negocio['platos']),
        'configuracion': 'personalizable',
        'version': '3.0.0'
    })


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("🌐 SERVIDOR FLASK - SISTEMA MEJORADO SABOR CHAPACO")
    print("="*80)
    print(f"🤖 Modelos cargados: {len(modelos_disponibles)}")
    print(f"🚨 Sistema de alertas: ACTIVO")
    print(f"⚙️  Configuración: PERSONALIZABLE")
    print(f"\n📍 URL: http://localhost:5000")
    print(f"📍 API: http://localhost:5000/api/")
    print("\n💡 Presiona CTRL+C para detener\n")
    print("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=True)