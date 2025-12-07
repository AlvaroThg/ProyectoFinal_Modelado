"""
Script de Entrenamiento Completo - Todos los Modelos
Genera los archivos .pkl necesarios para el sistema
Ejecutar antes de iniciar la aplicación
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 ENTRENAMIENTO DE MODELOS - SABOR CHAPACO")
print("="*80)

# ============================================================================
# 1. GENERAR DATASET SINTÉTICO REALISTA
# ============================================================================

def generar_dataset_restaurante(n_dias=365, platos=None):
    """
    Genera dataset sintético realista para restaurante
    """
    if platos is None:
        platos = ['Ranga Ranga', 'Sajta de Pollo', 'Chancho a la Cruz', 
                  'Picante de Lengua', 'Chicharrón de Cerdo']
    
    print(f"\n📊 Generando dataset: {n_dias} días x {len(platos)} platos")
    
    # Fechas
    fecha_inicio = datetime(2024, 1, 1)
    fechas = [fecha_inicio + timedelta(days=i) for i in range(n_dias)]
    
    datos = []
    np.random.seed(42)
    
    for plato in platos:
        # Demanda base por plato
        demanda_base = {
            'Ranga Ranga': 42,
            'Sajta de Pollo': 35,
            'Chancho a la Cruz': 28,
            'Picante de Lengua': 18,
            'Chicharrón de Cerdo': 15
        }
        
        base = demanda_base.get(plato, 30)
        
        for fecha in fechas:
            # Componentes de la demanda
            
            # 1. Tendencia (crecimiento leve)
            tendencia = base + (fecha - fecha_inicio).days * 0.01
            
            # 2. Estacionalidad semanal
            dia_semana = fecha.weekday()
            if dia_semana >= 4:  # Viernes-Domingo
                estacionalidad_semanal = base * 0.35
            else:
                estacionalidad_semanal = base * -0.10
            
            # 3. Estacionalidad mensual (temporada turística)
            mes = fecha.month
            if mes in [5, 6, 7, 8, 9, 10]:  # Mayo-Octubre
                estacionalidad_mensual = base * 0.28
            else:
                estacionalidad_mensual = base * -0.15
            
            # 4. Efecto clima (aleatorio)
            clima = np.random.choice(['Soleado', 'Nublado', 'Lluvioso'], p=[0.60, 0.30, 0.10])
            if clima == 'Lluvioso' and plato in ['Ranga Ranga', 'Sajta de Pollo']:
                efecto_clima = base * 0.15
            else:
                efecto_clima = 0
            
            # 5. Eventos (aleatorio)
            es_evento = np.random.random() < 0.10  # 10% días con evento
            if es_evento:
                evento = np.random.choice(['Festival', 'Fiesta', 'Feriado'])
                efecto_evento = base * 0.25
            else:
                evento = 'Ninguno'
                efecto_evento = 0
            
            # 6. Promociones (aleatorio)
            es_promocion = np.random.random() < 0.15  # 15% días con promo
            if es_promocion:
                promocion = np.random.choice(['2x1', 'Descuento', 'Menu del Dia'])
                efecto_promocion = base * 0.30
            else:
                promocion = 'Ninguna'
                efecto_promocion = 0
            
            # 7. Ruido aleatorio
            ruido = np.random.normal(0, base * 0.10)
            
            # Demanda total
            porciones = (tendencia + estacionalidad_semanal + estacionalidad_mensual + 
                        efecto_clima + efecto_evento + efecto_promocion + ruido)
            porciones = max(5, int(porciones))  # Mínimo 5 porciones
            
            # Hora de servicio (concentrado en horarios peak)
            hora = np.random.choice([15, 16, 17, 18, 19, 20, 21, 22, 23, 0], 
                                   p=[0.05, 0.05, 0.10, 0.15, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02])
            
            datos.append({
                'fecha': fecha,
                'plato': plato,
                'porciones_vendidas': porciones,
                'hora_servicio': hora,
                'dia_semana': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][dia_semana],
                'es_fin_semana': 1 if dia_semana >= 4 else 0,
                'condicion_climatica': clima,
                'evento_local': evento,
                'tipo_promocion': promocion,
                'mes': mes,
                'trimestre': (mes - 1) // 3 + 1
            })
    
    df = pd.DataFrame(datos)
    print(f"✅ Dataset generado: {len(df)} registros")
    print(f"📊 Rango porciones: {df['porciones_vendidas'].min()}-{df['porciones_vendidas'].max()}")
    
    return df


# ============================================================================
# 2. ENTRENAR MODELO ORIGINAL (POLINOMIAL)
# ============================================================================

def entrenar_modelo_polinomial(df):
    """Entrena el modelo polinomial original"""
    print("\n" + "="*80)
    print("🔵 MODELO 1: REGRESIÓN POLINÓMICA")
    print("="*80)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.pipeline import Pipeline
    import joblib
    
    # Preparar features
    df_model = df.copy()
    df_model['hora_num'] = df_model['hora_servicio']
    df_model['dia_semana_num'] = df_model['dia_semana'].map({
        'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
        'Viernes': 4, 'Sábado': 5, 'Domingo': 6
    })
    
    # Codificar variables categóricas
    label_encoders = {}
    for col in ['plato', 'condicion_climatica', 'evento_local', 'tipo_promocion']:
        le = LabelEncoder()
        df_model[f'{col}_encoded'] = le.fit_transform(df_model[col])
        label_encoders[col] = le
    
    # Features numéricas
    feature_cols = ['hora_num', 'es_fin_semana', 'mes', 'dia_semana_num', 'trimestre',
                    'plato_encoded', 'condicion_climatica_encoded', 
                    'evento_local_encoded', 'tipo_promocion_encoded']
    
    X = df_model[feature_cols]
    y = df_model['porciones_vendidas']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline: Scaling + Polynomial + Ridge
    modelo = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3)),
        ('ridge', Ridge(alpha=200.0))
    ])
    
    print("🔄 Entrenando modelo polinomial...")
    modelo.fit(X_train, y_train)
    
    # Evaluar
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"✅ R² Score: {r2:.4f}")
    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ RMSE: {rmse:.2f}")
    print(f"✅ MAPE: {mape:.2f}%")
    
    # Guardar
    joblib.dump(modelo, 'modelo_final_sabor_chapaco.pkl')
    joblib.dump(modelo.named_steps['scaler'], 'scaler_sabor_chapaco.pkl')
    joblib.dump(label_encoders, 'label_encoders_sabor_chapaco.pkl')
    joblib.dump(modelo.named_steps['poly'], 'polynomial_features.pkl')
    joblib.dump({
        'model_name': 'Regresión Polinómica Grado 3 (α=200.0)',
        'test_metrics': {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape},
        'is_polynomial': True,
        'polynomial_degree': 3,
        'best_params': {'alpha': 200.0},
        'feature_names': feature_cols,
        'trained_date': datetime.now().strftime('%Y-%m-%d')
    }, 'model_info.pkl')
    
    print("💾 Archivos guardados:")
    print("   - modelo_final_sabor_chapaco.pkl")
    print("   - scaler_sabor_chapaco.pkl")
    print("   - label_encoders_sabor_chapaco.pkl")
    print("   - polynomial_features.pkl")
    print("   - model_info.pkl")
    
    return modelo, {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# ============================================================================
# 3. ENTRENAR PROPHET
# ============================================================================

def entrenar_prophet(df, plato='Ranga Ranga'):
    """Entrena modelo Prophet"""
    print("\n" + "="*80)
    print("🔵 MODELO 2: PROPHET")
    print("="*80)
    
    try:
        from prophet import Prophet
        from sklearn.metrics import r2_score, mean_absolute_error
        import joblib
        
        # Filtrar por plato
        df_plato = df[df['plato'] == plato].copy()
        df_plato = df_plato.groupby('fecha')['porciones_vendidas'].sum().reset_index()
        
        # Formato Prophet
        df_prophet = pd.DataFrame({
            'ds': df_plato['fecha'],
            'y': df_plato['porciones_vendidas']
        })
        
        # Split
        split_idx = int(len(df_prophet) * 0.8)
        train = df_prophet[:split_idx]
        test = df_prophet[split_idx:]
        
        # Entrenar
        print(f"🔄 Entrenando Prophet para {plato}...")
        modelo = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        modelo.fit(train)
        
        # Evaluar
        forecast = modelo.predict(test)
        y_true = test['y'].values
        y_pred = forecast['yhat'].values[:len(y_true)]
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"✅ R² Score: {r2:.4f}")
        print(f"✅ MAE: {mae:.2f}")
        
        # Guardar
        joblib.dump({
            'modelo': modelo,
            'nombre': 'Prophet',
            'metricas': {'R2': r2, 'MAE': mae},
            'plato': plato,
            'entrenado': True
        }, 'modelo_prophet_sabor_chapaco.pkl')
        
        print("💾 Guardado: modelo_prophet_sabor_chapaco.pkl")
        
        return modelo, {'R2': r2, 'MAE': mae}
        
    except ImportError:
        print("⚠️  Prophet no instalado. Instalar con: pip install prophet")
        return None, None


# ============================================================================
# 4. ENTRENAR SARIMA
# ============================================================================

def entrenar_sarima(df, plato='Ranga Ranga'):
    """Entrena modelo SARIMA"""
    print("\n" + "="*80)
    print("🔵 MODELO 3: SARIMA")
    print("="*80)
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from sklearn.metrics import r2_score, mean_absolute_error
        import joblib
        
        # Preparar serie temporal
        df_plato = df[df['plato'] == plato].copy()
        df_plato = df_plato.groupby('fecha')['porciones_vendidas'].sum().reset_index()
        df_plato = df_plato.set_index('fecha')
        serie = df_plato['porciones_vendidas']
        
        # Split
        split_idx = int(len(serie) * 0.8)
        train = serie[:split_idx]
        test = serie[split_idx:]
        
        # Entrenar
        print(f"🔄 Entrenando SARIMA para {plato}...")
        modelo = SARIMAX(train, 
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 7))
        modelo_fit = modelo.fit(disp=False)
        
        # Evaluar
        forecast = modelo_fit.forecast(steps=len(test))
        r2 = r2_score(test.values, forecast.values)
        mae = mean_absolute_error(test.values, forecast.values)
        
        print(f"✅ R² Score: {r2:.4f}")
        print(f"✅ MAE: {mae:.2f}")
        
        # Guardar
        joblib.dump({
            'modelo': modelo_fit,
            'nombre': 'SARIMA',
            'metricas': {'R2': r2, 'MAE': mae},
            'plato': plato,
            'entrenado': True
        }, 'modelo_sarima_sabor_chapaco.pkl')
        
        print("💾 Guardado: modelo_sarima_sabor_chapaco.pkl")
        
        return modelo_fit, {'R2': r2, 'MAE': mae}
        
    except ImportError:
        print("⚠️  statsmodels no instalado. Instalar con: pip install statsmodels")
        return None, None


# ============================================================================
# 5. ENTRENAR XGBOOST
# ============================================================================

def entrenar_xgboost(df):
    """Entrena modelo XGBoost"""
    print("\n" + "="*80)
    print("🔵 MODELO 4: XGBOOST")
    print("="*80)
    
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import r2_score, mean_absolute_error
        import joblib
        
        # Preparar features
        df_xgb = df.copy()
        
        # Features temporales
        df_xgb['hora_num'] = df_xgb['hora_servicio']
        df_xgb['dia_semana_num'] = df_xgb['dia_semana'].map({
            'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
            'Viernes': 4, 'Sábado': 5, 'Domingo': 6
        })
        
        # Lags
        df_xgb = df_xgb.sort_values('fecha')
        for lag in [1, 7]:
            df_xgb[f'lag_{lag}'] = df_xgb.groupby('plato')['porciones_vendidas'].shift(lag)
        
        # Rolling mean
        df_xgb['rolling_mean_7'] = df_xgb.groupby('plato')['porciones_vendidas'].rolling(7).mean().reset_index(0, drop=True)
        
        df_xgb = df_xgb.dropna()
        
        # Codificar categóricas
        for col in ['plato', 'condicion_climatica', 'evento_local', 'tipo_promocion']:
            le = LabelEncoder()
            df_xgb[f'{col}_encoded'] = le.fit_transform(df_xgb[col])
        
        # Features finales
        feature_cols = ['hora_num', 'es_fin_semana', 'mes', 'dia_semana_num', 'trimestre',
                       'plato_encoded', 'condicion_climatica_encoded', 
                       'evento_local_encoded', 'tipo_promocion_encoded',
                       'lag_1', 'lag_7', 'rolling_mean_7']
        
        X = df_xgb[feature_cols]
        y = df_xgb['porciones_vendidas']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar
        print("🔄 Entrenando XGBoost...")
        modelo = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )
        modelo.fit(X_train, y_train)
        
        # Evaluar
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"✅ R² Score: {r2:.4f}")
        print(f"✅ MAE: {mae:.2f}")
        
        # Importancia de features
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n🔝 Top 5 Features:")
        print(importances.head().to_string(index=False))
        
        # Guardar
        joblib.dump({
            'modelo': modelo,
            'nombre': 'XGBoost',
            'metricas': {'R2': r2, 'MAE': mae},
            'feature_names': feature_cols,
            'feature_importance': importances,
            'entrenado': True
        }, 'modelo_xgboost_sabor_chapaco.pkl')
        
        print("\n💾 Guardado: modelo_xgboost_sabor_chapaco.pkl")
        
        return modelo, {'R2': r2, 'MAE': mae}
        
    except ImportError:
        print("⚠️  XGBoost no instalado. Instalar con: pip install xgboost")
        return None, None


# ============================================================================
# 6. ENTRENAR LSTM (OPCIONAL)
# ============================================================================

def entrenar_lstm(df, plato='Ranga Ranga'):
    """Entrena modelo LSTM"""
    print("\n" + "="*80)
    print("🔵 MODELO 5: LSTM (Opcional)")
    print("="*80)
    
    try:
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import r2_score, mean_absolute_error
        import joblib
        
        # Preparar serie temporal
        df_plato = df[df['plato'] == plato].copy()
        df_plato = df_plato.groupby('fecha')['porciones_vendidas'].sum().reset_index()
        serie = df_plato['porciones_vendidas'].values
        
        # Normalizar
        scaler = MinMaxScaler()
        serie_scaled = scaler.fit_transform(serie.reshape(-1, 1))
        
        # Crear secuencias
        ventana = 14
        X, y = [], []
        for i in range(ventana, len(serie_scaled)):
            X.append(serie_scaled[i-ventana:i, 0])
            y.append(serie_scaled[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Modelo
        print("🔄 Entrenando LSTM...")
        modelo = Sequential([
            LSTM(64, return_sequences=True, input_shape=(ventana, 1)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1)
        ])
        
        modelo.compile(optimizer='adam', loss='mse')
        modelo.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0,
                  validation_split=0.2)
        
        # Evaluar
        y_pred = modelo.predict(X_test, verbose=0)
        
        # Desnormalizar
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_real = scaler.inverse_transform(y_pred).flatten()
        
        r2 = r2_score(y_test_real, y_pred_real)
        mae = mean_absolute_error(y_test_real, y_pred_real)
        
        print(f"✅ R² Score: {r2:.4f}")
        print(f"✅ MAE: {mae:.2f}")
        
        # Guardar
        joblib.dump({
            'modelo': modelo,
            'scaler': scaler,
            'nombre': 'LSTM',
            'metricas': {'R2': r2, 'MAE': mae},
            'plato': plato,
            'ventana': ventana,
            'entrenado': True
        }, 'modelo_lstm_sabor_chapaco.pkl')
        
        print("💾 Guardado: modelo_lstm_sabor_chapaco.pkl")
        
        return modelo, {'R2': r2, 'MAE': mae}
        
    except ImportError:
        print("⚠️  TensorFlow no instalado. Instalar con: pip install tensorflow")
        return None, None


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Ejecuta entrenamiento completo"""
    
    print("\n🎯 INICIO DEL PROCESO DE ENTRENAMIENTO\n")
    
    # 1. Generar dataset
    df = generar_dataset_restaurante(n_dias=365)
    
    # Guardar dataset
    df.to_csv('dataset_sabor_chapaco.csv', index=False)
    print(f"\n💾 Dataset guardado: dataset_sabor_chapaco.csv")
    
    # 2. Entrenar todos los modelos
    resultados = {}
    
    # Modelo 1: Polinomial (OBLIGATORIO)
    modelo_poly, metricas_poly = entrenar_modelo_polinomial(df)
    resultados['Polinomial'] = metricas_poly
    
    # Modelo 2: Prophet (OPCIONAL)
    modelo_prophet, metricas_prophet = entrenar_prophet(df)
    if metricas_prophet:
        resultados['Prophet'] = metricas_prophet
    
    # Modelo 3: SARIMA (OPCIONAL)
    modelo_sarima, metricas_sarima = entrenar_sarima(df)
    if metricas_sarima:
        resultados['SARIMA'] = metricas_sarima
    
    # Modelo 4: XGBoost (RECOMENDADO)
    modelo_xgb, metricas_xgb = entrenar_xgboost(df)
    if metricas_xgb:
        resultados['XGBoost'] = metricas_xgb
    
    # Modelo 5: LSTM (OPCIONAL - requiere TensorFlow)
    modelo_lstm, metricas_lstm = entrenar_lstm(df)
    if metricas_lstm:
        resultados['LSTM'] = metricas_lstm
    
    # 3. Comparativa final
    print("\n" + "="*80)
    print("📊 RESUMEN COMPARATIVO DE MODELOS")
    print("="*80)
    
    df_comparativa = pd.DataFrame(resultados).T
    df_comparativa = df_comparativa.sort_values('MAE')
    print(df_comparativa.to_string())
    
    print("\n🏆 Mejor modelo (menor MAE):", df_comparativa.index[0])
    
    print("\n✨ ¡ENTRENAMIENTO COMPLETADO!")
    print("\n📁 Archivos generados:")
    print("   ✅ modelo_final_sabor_chapaco.pkl")
    print("   ✅ scaler_sabor_chapaco.pkl")
    print("   ✅ label_encoders_sabor_chapaco.pkl")
    print("   ✅ polynomial_features.pkl")
    print("   ✅ model_info.pkl")
    print("   ✅ dataset_sabor_chapaco.csv")
    
    if metricas_prophet:
        print("   ✅ modelo_prophet_sabor_chapaco.pkl")
    if metricas_sarima:
        print("   ✅ modelo_sarima_sabor_chapaco.pkl")
    if metricas_xgb:
        print("   ✅ modelo_xgboost_sabor_chapaco.pkl")
    if metricas_lstm:
        print("   ✅ modelo_lstm_sabor_chapaco.pkl")
    
    print("\n🚀 Ahora puedes ejecutar: python app_mejorado.py")


if __name__ == "__main__":
    main()