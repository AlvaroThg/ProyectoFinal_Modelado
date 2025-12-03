# Sistema Predictivo de Demanda - Restaurante Sabor Chapaco

Sistema web basado en Machine Learning para predecir la demanda de platos en el restaurante Sabor Chapaco.

## 🚀 Inicio Rápido

1. **DESCOMPRIMIR EL ARCHIVO ZIP**

2. **Crear y activar entorno virtual**
   ```powershell
   # En PowerShell
   python -m venv .venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias**
   ```powershell
   pip install -r requerimientos.txt
   ```

4. **Estructura de archivos estáticos**
   ```
   static/
   ├── img/
   │   └── SaborChapaco.jpeg    # Logo del restaurante
   └── graficos/
       ├── 01_pred_vs_real.png  # Gráfico predicción vs real
       └── ...                  # Otros gráficos de métricas
   ```

   > 📝 **Importante**: Todos los assets estáticos (imágenes, gráficos) deben colocarse en la carpeta `static/`. 
   
   > - Imágenes generales en `static/img/`
   > - Gráficos de métricas en `static/graficos/`

   Nota: Antes de correr app.py, se debe correr por separado el visualizaciones.py

   ```powershell
   python visualizaciones.py
   ```

5. **Ejecutar el servidor**
   ```powershell
   python app.py
   ```
   El servidor estará disponible en http://localhost:5000

## 📊 Endpoints API

- `POST /api/predict` - Realizar predicción de demanda
- `GET /api/model-info` - Información del modelo
- `GET /api/health` - Estado del servidor
- `GET /api/platos` - Lista de platos disponibles
- `GET /api/statistics` - Estadísticas del sistema

## 🔧 Tecnologías

- **Backend**: Flask, scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: Regresión (ver detalles en `model_info.pkl`)

## 📁 Archivos del Proyecto

- `app.py` - Servidor Flask y lógica del backend
- `index.html` - Interfaz web
- `requerimientos.txt` - Dependencias del proyecto
- `modelo_final_sabor_chapaco.pkl` - Modelo entrenado
- `scaler_sabor_chapaco.pkl` - Scaler para features
- `label_encoders_sabor_chapaco.pkl` - Encoders para variables categóricas
- `model_info.pkl` - Información y métricas del modelo
- `polynomial_features.pkl` - Features polinómicas (si aplica)

## ⚙️ Configuración

1. **Modo Debug**
   En `app.py`, configura `debug=True` para desarrollo o `debug=False` para producción:
   ```python
   app.run(host='0.0.0.0', port=5000, debug=True) 
   ```

2. **Referencias a archivos estáticos**
   - En HTML: `<img src="/static/img/SaborChapaco.jpeg">`
   - En CSS: `url('/static/img/background.jpg')`

## 📈 Métricas del Modelo

- R² Score: Ver `/api/model-info`
- MAE: Error absoluto medio en porciones
- RMSE: Error cuadrático medio
- MAPE: Error porcentual absoluto medio

