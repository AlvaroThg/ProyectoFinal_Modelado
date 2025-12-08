"""
Script para Corregir el Modelo - Features Compatibles
Genera modelos compatibles con tu app.py actual
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔧 CORRIGIENDO MODELO - FEATURES COMPATIBLES")
print("="*80)

# ============================================================================
# 1. GENERAR DATASET CON FEATURES CORRECTAS
# ============================================================================

def generar_dataset():
    """Genera dataset sintético"""
    print("\n📊 Generando dataset...")
    
    fecha_inicio = datetime(2024, 1, 1)
    n_dias = 365
    fechas = [fecha_inicio + timedelta(days=i) for i in range(n_dias)]
    
    platos = ['Ranga Ranga', 'Sajta de Pollo', 'Chancho a la Cruz', 
              'Picante de Lengua', 'Chicharrón de Cerdo']
    
    datos = []
    np.random.seed(42)
    
    for plato in platos:
        demanda_base = {
            'Ranga Ranga': 42, 'Sajta de Pollo': 35,
            'Chancho a la Cruz': 28, 'Picante de Lengua': 18,
            'Chicharrón de Cerdo': 15
        }
        base = demanda_base[plato]
        
        for fecha in fechas:
            dia_semana = fecha.weekday()
            mes = fecha.month
            
            # Calcular demanda
            tendencia = base + (fecha - fecha_inicio).days * 0.01
            
            if dia_semana >= 4:
                est_semanal = base * 0.35
            else:
                est_semanal = base * -0.10
            
            if mes in [5,6,7,8,9,10]:
                est_mensual = base * 0.28
            else:
                est_mensual = base * -0.15
            
            clima = np.random.choice(['Soleado', 'Nublado', 'Lluvioso'], 
                                   p=[0.60, 0.30, 0.10])
            if clima == 'Lluvioso' and plato in ['Ranga Ranga', 'Sajta de Pollo']:
                efecto_clima = base * 0.15
            else:
                efecto_clima = 0
            
            es_evento = np.random.random() < 0.10
            if es_evento:
                evento = np.random.choice(['Festival', 'Fiesta', 'Feriado'])
                efecto_evento = base * 0.25
            else:
                evento = 'Ninguno'
                efecto_evento = 0
            
            es_promo = np.random.random() < 0.15
            if es_promo:
                promocion = np.random.choice(['2x1', 'Descuento', 'Menu del Dia'])
                efecto_promo = base * 0.30
            else:
                promocion = 'Ninguna'
                efecto_promo = 0
            
            ruido = np.random.normal(0, base * 0.10)
            
            porciones = (tendencia + est_semanal + est_mensual + 
                        efecto_clima + efecto_evento + efecto_promo + ruido)
            porciones = max(5, int(porciones))
            
            hora = np.random.choice([15,16,17,18,19,20,21,22,23,0],
                                   p=[0.05,0.05,0.10,0.15,0.25,0.20,0.10,0.05,0.03,0.02])
            
            datos.append({
                'fecha': fecha,
                'plato': plato,
                'porciones_vendidas': porciones,
                'hora_servicio': hora,
                'dia_semana': ['Lunes','Martes','Miércoles','Jueves',
                              'Viernes','Sábado','Domingo'][dia_semana],
                'es_fin_semana': 1 if dia_semana >= 4 else 0,
                'condicion_climatica': clima,
                'evento_local': evento,
                'tipo_promocion': promocion,
                'mes': mes,
                'trimestre': (mes - 1) // 3 + 1
            })
    
    df = pd.DataFrame(datos)
    print(f"✅ Dataset generado: {len(df)} registros")
    return df


# ============================================================================
# 2. FEATURE ENGINEERING (IGUAL A TU APP.PY)
# ============================================================================

def crear_features(df):
    """Crea EXACTAMENTE las mismas features que usa tu app.py"""
    print("\n🔧 Creando features...")
    
    df['hora_num'] = df['hora_servicio']
    df['minuto'] = 0
    
    # Features de franja horaria
    df['es_almuerzo'] = ((df['hora_num'] >= 11) & (df['hora_num'] <= 15)).astype(int)
    df['es_cena'] = ((df['hora_num'] >= 18) & (df['hora_num'] <= 22)).astype(int)
    df['es_desayuno'] = ((df['hora_num'] >= 6) & (df['hora_num'] <= 10)).astype(int)
    
    # Features temporales
    df['es_fin_mes'] = (df['fecha'].dt.day > 25).astype(int)
    df['temporada_turistica'] = df['mes'].apply(lambda x: 1 if x in [5,6,7,8,9,10] else 0)
    
    # Features cíclicas
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['hora_sin'] = np.sin(2 * np.pi * df['hora_num'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora_num'] / 24)
    
    # Día de la semana numérico
    dias_map = {
        'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
        'Viernes': 4, 'Sábado': 5, 'Domingo': 6
    }
    df['dia_semana_num'] = df['dia_semana'].map(dias_map)
    
    return df


# ============================================================================
# 3. ENTRENAR MODELO CON FEATURES CORRECTAS
# ============================================================================

def entrenar_modelo_compatible(df):
    """Entrena modelo compatible con app.py"""
    print("\n🤖 Entrenando modelo compatible...")
    
    # Crear features
    df = crear_features(df)
    
    # Codificar categóricas
    label_encoders = {}
    for col in ['plato', 'condicion_climatica', 'evento_local', 'tipo_promocion']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # LISTA EXACTA DE FEATURES QUE USA TU APP.PY
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
    
    print(f"📊 Features totales: {len(feature_cols)}")
    print(f"   - Numéricas: {len(numeric_features)}")
    print(f"   - Categóricas: {len(categorical_encoded)}")
    
    # Preparar X, y
    X = df[feature_cols]
    y = df['porciones_vendidas']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar Pipeline completo
    print("\n🔄 Entrenando pipeline...")
    
    # Crear scaler independiente primero
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Crear polynomial features
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    # Entrenar Ridge
    ridge = Ridge(alpha=200.0)
    ridge.fit(X_train_poly, y_train)
    
    # Crear modelo completo (Pipeline manual)
    modelo_completo = Pipeline([
        ('scaler', scaler),
        ('poly', poly),
        ('ridge', ridge)
    ])
    
    # Evaluar
    y_pred = ridge.predict(X_test_poly)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n📊 MÉTRICAS:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Guardar archivos
    print("\n💾 Guardando archivos...")
    
    joblib.dump(modelo_completo, 'modelo_final_sabor_chapaco.pkl')
    print("   ✅ modelo_final_sabor_chapaco.pkl")
    
    joblib.dump(scaler, 'scaler_sabor_chapaco.pkl')
    print("   ✅ scaler_sabor_chapaco.pkl")
    
    joblib.dump(label_encoders, 'label_encoders_sabor_chapaco.pkl')
    print("   ✅ label_encoders_sabor_chapaco.pkl")
    
    joblib.dump(poly, 'polynomial_features.pkl')
    print("   ✅ polynomial_features.pkl")
    
    model_info = {
        'model_name': 'Regresión Polinómica Grado 3 (α=200.0)',
        'test_metrics': {
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        },
        'is_polynomial': True,
        'polynomial_degree': 3,
        'best_params': {'alpha': 200.0},
        'feature_names': feature_cols,
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_info, 'model_info.pkl')
    print("   ✅ model_info.pkl")
    
    return modelo_completo, model_info


# ============================================================================
# 4. VERIFICAR COMPATIBILIDAD
# ============================================================================

def verificar_modelo():
    """Verifica que el modelo funcione con app.py"""
    print("\n🔍 VERIFICANDO COMPATIBILIDAD...")
    
    try:
        # Cargar modelo
        modelo = joblib.load('modelo_final_sabor_chapaco.pkl')
        scaler = joblib.load('scaler_sabor_chapaco.pkl')
        label_encoders = joblib.load('label_encoders_sabor_chapaco.pkl')
        
        # Simular predicción como en app.py
        test_data = {
            'fecha': '2025-12-08',
            'plato': 'Ranga Ranga',
            'hora_servicio': 19,
            'dia_semana': 'Lunes',
            'condicion_climatica': 'Soleado',
            'evento_local': 'Ninguno',
            'tipo_promocion': 'Ninguna',
            'es_fin_semana': 0
        }
        
        # Crear features
        df_test = pd.DataFrame([test_data])
        df_test['fecha'] = pd.to_datetime(df_test['fecha'])
        df_test['mes'] = df_test['fecha'].dt.month
        df_test['dia'] = df_test['fecha'].dt.day
        df_test['hora_num'] = df_test['hora_servicio'].astype(int)
        df_test['minuto'] = 0
        
        df_test['es_almuerzo'] = ((df_test['hora_num'] >= 11) & (df_test['hora_num'] <= 15)).astype(int)
        df_test['es_cena'] = ((df_test['hora_num'] >= 18) & (df_test['hora_num'] <= 22)).astype(int)
        df_test['es_desayuno'] = ((df_test['hora_num'] >= 6) & (df_test['hora_num'] <= 10)).astype(int)
        
        df_test['trimestre'] = df_test['mes'].apply(lambda x: (x-1)//3 + 1)
        df_test['es_fin_mes'] = (df_test['fecha'].dt.day > 25).astype(int)
        df_test['temporada_turistica'] = df_test['mes'].apply(lambda x: 1 if x in [5,6,7,8,9,10] else 0)
        
        df_test['mes_sin'] = np.sin(2 * np.pi * df_test['mes'] / 12)
        df_test['mes_cos'] = np.cos(2 * np.pi * df_test['mes'] / 12)
        df_test['hora_sin'] = np.sin(2 * np.pi * df_test['hora_num'] / 24)
        df_test['hora_cos'] = np.cos(2 * np.pi * df_test['hora_num'] / 24)
        
        dias_map = {
            'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
            'Viernes': 4, 'Sábado': 5, 'Domingo': 6
        }
        df_test['dia_semana_num'] = df_test['dia_semana'].map(dias_map)
        
        # Codificar categóricas
        for col in ['plato', 'condicion_climatica', 'evento_local', 'tipo_promocion']:
            le = label_encoders[col]
            df_test[f'{col}_encoded'] = le.transform(df_test[col].astype(str))
        
        # Features finales
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
        X_test = df_test[feature_cols]
        
        # Predecir
        prediccion = modelo.predict(X_test)[0]
        
        print(f"✅ Predicción exitosa: {prediccion:.1f} porciones")
        print(f"✅ El modelo es COMPATIBLE con app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n🎯 INICIANDO CORRECCIÓN DE MODELO\n")
    
    # 1. Generar dataset
    df = generar_dataset()
    
    # 2. Entrenar modelo compatible
    modelo, info = entrenar_modelo_compatible(df)
    
    # 3. Verificar
    if verificar_modelo():
        print("\n" + "="*80)
        print("✨ ¡ÉXITO! MODELO CORREGIDO Y VERIFICADO")
        print("="*80)
        print("\n🚀 Ahora puedes ejecutar: python app.py")
        print("   El servidor debería funcionar sin errores\n")
    else:
        print("\n❌ Hubo un problema. Revisa los errores arriba.\n")


if __name__ == "__main__":
    main()  