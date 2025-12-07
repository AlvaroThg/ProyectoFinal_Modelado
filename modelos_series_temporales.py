"""
Sistema de Modelos de Series Temporales - Sabor Chapaco
Implementa: Prophet, ARIMA/SARIMA, XGBoost, LSTM
Examen Final - Modelado y Simulación de Sistemas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Modelos de Series Temporales
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

# Deep Learning
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================================
# CLASE BASE PARA MODELOS
# ============================================================================

class ModeloSerieTemporalBase:
    """Clase base para todos los modelos de series temporales"""
    
    def __init__(self, nombre_modelo):
        self.nombre_modelo = nombre_modelo
        self.modelo = None
        self.metricas = {}
        self.entrenado = False
    
    def calcular_metricas(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        self.metricas = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        return self.metricas
    
    def guardar_modelo(self, ruta):
        """Guarda el modelo entrenado"""
        joblib.dump({
            'modelo': self.modelo,
            'nombre': self.nombre_modelo,
            'metricas': self.metricas,
            'entrenado': self.entrenado
        }, ruta)
        print(f"✅ Modelo {self.nombre_modelo} guardado en {ruta}")


# ============================================================================
# MODELO 1: PROPHET (Facebook)
# ============================================================================

class ModeloProphet(ModeloSerieTemporalBase):
    """
    Modelo Prophet de Facebook para series temporales
    Ventajas: Maneja estacionalidad, holidays, tendencias
    """
    
    def __init__(self):
        super().__init__("Prophet")
        self.modelo = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
    
    def preparar_datos(self, df, fecha_col='fecha', target_col='porciones_vendidas'):
        """Prepara datos en formato Prophet (ds, y)"""
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(df[fecha_col]),
            'y': df[target_col]
        })
        return df_prophet
    
    def agregar_regresores(self, df, columnas_adicionales):
        """Agrega variables adicionales como regresores"""
        for col in columnas_adicionales:
            self.modelo.add_regressor(col)
        return df
    
    def entrenar(self, df_train):
        """Entrena el modelo Prophet"""
        print(f"🔄 Entrenando {self.nombre_modelo}...")
        self.modelo.fit(df_train)
        self.entrenado = True
        print(f"✅ {self.nombre_modelo} entrenado exitosamente")
    
    def predecir(self, periodos_futuros=30, freq='D'):
        """Genera predicciones futuras"""
        future = self.modelo.make_future_dataframe(periods=periodos_futuros, freq=freq)
        forecast = self.modelo.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def evaluar(self, df_test):
        """Evalua el modelo en datos de test"""
        forecast = self.modelo.predict(df_test)
        y_true = df_test['y'].values
        y_pred = forecast['yhat'].values
        return self.calcular_metricas(y_true, y_pred)


# ============================================================================
# MODELO 2: ARIMA / SARIMA
# ============================================================================

class ModeloARIMA(ModeloSerieTemporalBase):
    """
    Modelo ARIMA/SARIMA para series temporales
    Ventajas: Captura autocorrelación y estacionalidad
    """
    
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,7), usar_sarima=True):
        nombre = "SARIMA" if usar_sarima else "ARIMA"
        super().__init__(nombre)
        self.order = order
        self.seasonal_order = seasonal_order if usar_sarima else None
        self.usar_sarima = usar_sarima
    
    def buscar_mejor_orden(self, serie, max_p=3, max_d=2, max_q=3):
        """Búsqueda de mejores parámetros (p,d,q) usando AIC"""
        print(f"🔍 Buscando mejores parámetros para {self.nombre_modelo}...")
        
        mejor_aic = np.inf
        mejor_orden = None
        
        for p in range(0, max_p + 1):
            for d in range(0, max_d + 1):
                for q in range(0, max_q + 1):
                    try:
                        if self.usar_sarima:
                            modelo_temp = SARIMAX(serie, 
                                                order=(p,d,q),
                                                seasonal_order=self.seasonal_order)
                        else:
                            modelo_temp = ARIMA(serie, order=(p,d,q))
                        
                        resultado = modelo_temp.fit(disp=False)
                        
                        if resultado.aic < mejor_aic:
                            mejor_aic = resultado.aic
                            mejor_orden = (p,d,q)
                    except:
                        continue
        
        self.order = mejor_orden
        print(f"✅ Mejor orden encontrado: {mejor_orden} (AIC: {mejor_aic:.2f})")
        return mejor_orden
    
    def entrenar(self, serie):
        """Entrena el modelo ARIMA/SARIMA"""
        print(f"🔄 Entrenando {self.nombre_modelo} con orden {self.order}...")
        
        if self.usar_sarima:
            self.modelo = SARIMAX(serie, 
                                 order=self.order,
                                 seasonal_order=self.seasonal_order)
        else:
            self.modelo = ARIMA(serie, order=self.order)
        
        self.modelo_fit = self.modelo.fit(disp=False)
        self.entrenado = True
        print(f"✅ {self.nombre_modelo} entrenado exitosamente")
    
    def predecir(self, pasos=30):
        """Genera predicciones futuras"""
        forecast = self.modelo_fit.forecast(steps=pasos)
        return forecast
    
    def evaluar(self, serie_test):
        """Evalúa el modelo en datos de test"""
        n_test = len(serie_test)
        forecast = self.predecir(pasos=n_test)
        return self.calcular_metricas(serie_test.values, forecast.values)


# ============================================================================
# MODELO 3: XGBOOST REGRESSOR
# ============================================================================

class ModeloXGBoost(ModeloSerieTemporalBase):
    """
    XGBoost para series temporales con features temporales
    Ventajas: Captura relaciones no-lineales, rápido, robusto
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        super().__init__("XGBoost Regressor")
        self.modelo = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=42
        )
        self.feature_names = []
    
    def crear_features_temporales(self, df, fecha_col='fecha', target_col='porciones_vendidas'):
        """
        Crea features temporales para XGBoost:
        - Lags (valores anteriores)
        - Rolling statistics (medias móviles)
        - Variables temporales (hora, día, mes, etc.)
        """
        df = df.copy()
        df[fecha_col] = pd.to_datetime(df[fecha_col])
        df = df.sort_values(fecha_col)
        
        # Variables temporales básicas
        df['hora'] = df[fecha_col].dt.hour
        df['dia_semana'] = df[fecha_col].dt.dayofweek
        df['dia_mes'] = df[fecha_col].dt.day
        df['mes'] = df[fecha_col].dt.month
        df['trimestre'] = df[fecha_col].dt.quarter
        df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
        
        # Lags (valores pasados)
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics (ventanas móviles)
        for window in [3, 7, 14]:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        
        # Diferencias
        df['diff_1'] = df[target_col].diff(1)
        df['diff_7'] = df[target_col].diff(7)
        
        # Eliminar filas con NaN (por lags y rolling)
        df = df.dropna()
        
        self.feature_names = [col for col in df.columns if col not in [fecha_col, target_col]]
        
        return df
    
    def entrenar(self, X_train, y_train):
        """Entrena el modelo XGBoost"""
        print(f"🔄 Entrenando {self.nombre_modelo}...")
        print(f"📊 Features utilizados: {len(self.feature_names)}")
        
        self.modelo.fit(X_train, y_train,
                       eval_set=[(X_train, y_train)],
                       verbose=False)
        
        self.entrenado = True
        print(f"✅ {self.nombre_modelo} entrenado exitosamente")
    
    def predecir(self, X_test):
        """Genera predicciones"""
        return self.modelo.predict(X_test)
    
    def evaluar(self, X_test, y_test):
        """Evalúa el modelo"""
        y_pred = self.predecir(X_test)
        return self.calcular_metricas(y_test, y_pred)
    
    def obtener_importancia_features(self):
        """Retorna importancia de features"""
        importances = self.modelo.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance


# ============================================================================
# MODELO 4: LSTM (Deep Learning)
# ============================================================================

class ModeloLSTM(ModeloSerieTemporalBase):
    """
    Red Neuronal LSTM para series temporales
    Ventajas: Captura dependencias de largo plazo
    """
    
    def __init__(self, ventana_temporal=7, unidades_lstm=50, capas=2):
        super().__init__("LSTM")
        self.ventana_temporal = ventana_temporal
        self.unidades_lstm = unidades_lstm
        self.capas = capas
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def preparar_secuencias(self, serie):
        """
        Prepara secuencias para LSTM
        Ventana deslizante: [t-n, t-n+1, ..., t-1] -> t
        """
        # Normalizar datos
        serie_scaled = self.scaler.fit_transform(serie.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.ventana_temporal, len(serie_scaled)):
            X.append(serie_scaled[i-self.ventana_temporal:i, 0])
            y.append(serie_scaled[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape para LSTM: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def construir_modelo(self, input_shape):
        """Construye arquitectura LSTM"""
        self.modelo = Sequential()
        
        # Primera capa LSTM
        self.modelo.add(LSTM(units=self.unidades_lstm, 
                           return_sequences=(self.capas > 1),
                           input_shape=input_shape))
        self.modelo.add(Dropout(0.2))
        
        # Capas adicionales
        for i in range(1, self.capas):
            return_seq = (i < self.capas - 1)
            self.modelo.add(LSTM(units=self.unidades_lstm, 
                               return_sequences=return_seq))
            self.modelo.add(Dropout(0.2))
        
        # Capa de salida
        self.modelo.add(Dense(units=1))
        
        # Compilar
        self.modelo.compile(optimizer='adam', loss='mean_squared_error')
        
        print(f"📐 Arquitectura LSTM creada:")
        self.modelo.summary()
    
    def entrenar(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Entrena el modelo LSTM"""
        print(f"🔄 Entrenando {self.nombre_modelo}...")
        
        history = self.modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        self.entrenado = True
        self.history = history
        print(f"✅ {self.nombre_modelo} entrenado exitosamente")
        
        return history
    
    def predecir(self, X_test):
        """Genera predicciones y desnormaliza"""
        predicciones_scaled = self.modelo.predict(X_test)
        predicciones = self.scaler.inverse_transform(predicciones_scaled)
        return predicciones.flatten()
    
    def evaluar(self, X_test, y_test):
        """Evalúa el modelo"""
        # Desnormalizar y_test
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Predecir
        y_pred = self.predecir(X_test)
        
        return self.calcular_metricas(y_test_original, y_pred)


# ============================================================================
# ENSEMBLE: COMBINACIÓN DE MODELOS
# ============================================================================

class EnsembleModelos:
    """
    Combina predicciones de múltiples modelos
    Estrategias: promedio simple, promedio ponderado, stacking
    """
    
    def __init__(self, modelos, pesos=None):
        self.modelos = modelos
        self.pesos = pesos if pesos else [1/len(modelos)] * len(modelos)
        self.nombre = "Ensemble"
    
    def predecir_promedio_simple(self, *args):
        """Promedio simple de predicciones"""
        predicciones = []
        
        for modelo in self.modelos:
            if hasattr(modelo, 'predecir'):
                pred = modelo.predecir(*args)
                predicciones.append(pred)
        
        return np.mean(predicciones, axis=0)
    
    def predecir_promedio_ponderado(self, *args):
        """Promedio ponderado por métricas de cada modelo"""
        predicciones = []
        
        for modelo, peso in zip(self.modelos, self.pesos):
            if hasattr(modelo, 'predecir'):
                pred = modelo.predecir(*args)
                predicciones.append(pred * peso)
        
        return np.sum(predicciones, axis=0)


# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================

def entrenar_todos_los_modelos(df, fecha_col='fecha', target_col='porciones_vendidas', 
                              plato=None, test_size=0.2):
    """
    Entrena todos los modelos de series temporales
    """
    
    print("="*80)
    print("🚀 ENTRENAMIENTO DE MODELOS DE SERIES TEMPORALES")
    print("="*80)
    
    if plato:
        df = df[df['plato'] == plato].copy()
        print(f"📊 Filtrando datos para: {plato}")
    
    df = df.sort_values(fecha_col)
    serie_completa = df[target_col]
    
    # Split train/test
    split_idx = int(len(df) * (1 - test_size))
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    print(f"📊 Total datos: {len(df)}")
    print(f"📊 Train: {len(train_df)} | Test: {len(test_df)}")
    print()
    
    resultados = {}
    
    # ========== MODELO 1: PROPHET ==========
    print("\n" + "="*80)
    print("1️⃣ PROPHET")
    print("="*80)
    
    modelo_prophet = ModeloProphet()
    df_prophet_train = modelo_prophet.preparar_datos(train_df, fecha_col, target_col)
    df_prophet_test = modelo_prophet.preparar_datos(test_df, fecha_col, target_col)
    
    modelo_prophet.entrenar(df_prophet_train)
    metricas_prophet = modelo_prophet.evaluar(df_prophet_test)
    
    print(f"📊 Métricas Prophet: MAE={metricas_prophet['MAE']:.2f}, R²={metricas_prophet['R2']:.4f}")
    resultados['Prophet'] = {
        'modelo': modelo_prophet,
        'metricas': metricas_prophet
    }
    
    # ========== MODELO 2: SARIMA ==========
    print("\n" + "="*80)
    print("2️⃣ SARIMA")
    print("="*80)
    
    modelo_sarima = ModeloARIMA(order=(1,1,1), seasonal_order=(1,1,1,7), usar_sarima=True)
    modelo_sarima.buscar_mejor_orden(train_df[target_col], max_p=2, max_d=1, max_q=2)
    modelo_sarima.entrenar(train_df[target_col])
    metricas_sarima = modelo_sarima.evaluar(test_df[target_col])
    
    print(f"📊 Métricas SARIMA: MAE={metricas_sarima['MAE']:.2f}, R²={metricas_sarima['R2']:.4f}")
    resultados['SARIMA'] = {
        'modelo': modelo_sarima,
        'metricas': metricas_sarima
    }
    
    # ========== MODELO 3: XGBOOST ==========
    print("\n" + "="*80)
    print("3️⃣ XGBOOST")
    print("="*80)
    
    modelo_xgb = ModeloXGBoost(n_estimators=200, max_depth=8, learning_rate=0.05)
    
    # Preparar datos con features temporales
    df_xgb = modelo_xgb.crear_features_temporales(df, fecha_col, target_col)
    split_idx_xgb = int(len(df_xgb) * (1 - test_size))
    
    X = df_xgb[modelo_xgb.feature_names]
    y = df_xgb[target_col]
    
    X_train, X_test = X[:split_idx_xgb], X[split_idx_xgb:]
    y_train, y_test = y[:split_idx_xgb], y[split_idx_xgb:]
    
    modelo_xgb.entrenar(X_train, y_train)
    metricas_xgb = modelo_xgb.evaluar(X_test, y_test)
    
    print(f"📊 Métricas XGBoost: MAE={metricas_xgb['MAE']:.2f}, R²={metricas_xgb['R2']:.4f}")
    
    # Importancia de features
    importancia = modelo_xgb.obtener_importancia_features()
    print(f"\n🔝 Top 5 Features más importantes:")
    print(importancia.head())
    
    resultados['XGBoost'] = {
        'modelo': modelo_xgb,
        'metricas': metricas_xgb,
        'feature_importance': importancia
    }
    
    # ========== MODELO 4: LSTM (OPCIONAL) ==========
    print("\n" + "="*80)
    print("4️⃣ LSTM (Opcional)")
    print("="*80)
    
    try:
        modelo_lstm = ModeloLSTM(ventana_temporal=14, unidades_lstm=64, capas=2)
        
        X_lstm, y_lstm = modelo_lstm.preparar_secuencias(serie_completa)
        split_idx_lstm = int(len(X_lstm) * (1 - test_size))
        
        X_train_lstm = X_lstm[:split_idx_lstm]
        y_train_lstm = y_lstm[:split_idx_lstm]
        X_test_lstm = X_lstm[split_idx_lstm:]
        y_test_lstm = y_lstm[split_idx_lstm:]
        
        modelo_lstm.construir_modelo(input_shape=(X_train_lstm.shape[1], 1))
        modelo_lstm.entrenar(X_train_lstm, y_train_lstm, epochs=30, batch_size=16)
        metricas_lstm = modelo_lstm.evaluar(X_test_lstm, y_test_lstm)
        
        print(f"📊 Métricas LSTM: MAE={metricas_lstm['MAE']:.2f}, R²={metricas_lstm['R2']:.4f}")
        
        resultados['LSTM'] = {
            'modelo': modelo_lstm,
            'metricas': metricas_lstm
        }
    except Exception as e:
        print(f"⚠️ LSTM no disponible o error: {e}")
        print("   (Opcional - no afecta otros modelos)")
    
    # ========== COMPARATIVA FINAL ==========
    print("\n" + "="*80)
    print("📊 COMPARATIVA DE MODELOS")
    print("="*80)
    
    df_comparativa = pd.DataFrame({
        'Modelo': list(resultados.keys()),
        'MAE': [resultados[m]['metricas']['MAE'] for m in resultados],
        'RMSE': [resultados[m]['metricas']['RMSE'] for m in resultados],
        'R²': [resultados[m]['metricas']['R2'] for m in resultados],
        'MAPE': [resultados[m]['metricas']['MAPE'] for m in resultados]
    }).sort_values('MAE')
    
    print(df_comparativa.to_string(index=False))
    
    print("\n🏆 Mejor modelo (menor MAE):", df_comparativa.iloc[0]['Modelo'])
    
    return resultados, df_comparativa


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    # Generar datos de ejemplo
    print("📥 Generando datos de ejemplo...")
    
    fechas = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Simulación de demanda realista con estacionalidad
    tendencia = np.linspace(30, 50, len(fechas))
    estacionalidad_semanal = 10 * np.sin(2 * np.pi * np.arange(len(fechas)) / 7)
    estacionalidad_mensual = 5 * np.cos(2 * np.pi * np.arange(len(fechas)) / 30)
    ruido = np.random.normal(0, 3, len(fechas))
    
    porciones = tendencia + estacionalidad_semanal + estacionalidad_mensual + ruido
    porciones = np.maximum(porciones, 10)  # Mínimo 10 porciones
    
    df_ejemplo = pd.DataFrame({
        'fecha': fechas,
        'porciones_vendidas': porciones,
        'plato': 'Ranga Ranga'
    })
    
    print(f"✅ Dataset generado: {len(df_ejemplo)} registros")
    print(f"📊 Rango de ventas: {porciones.min():.0f} - {porciones.max():.0f} porciones")
    print()
    
    # Entrenar todos los modelos
    resultados, comparativa = entrenar_todos_los_modelos(
        df_ejemplo,
        fecha_col='fecha',
        target_col='porciones_vendidas',
        plato='Ranga Ranga',
        test_size=0.2
    )
    
    # Guardar modelos
    print("\n💾 Guardando modelos...")
    for nombre, resultado in resultados.items():
        ruta = f'modelo_{nombre.lower()}_sabor_chapaco.pkl'
        resultado['modelo'].guardar_modelo(ruta)
    
    print("\n✨ ¡Proceso completado exitosamente!")