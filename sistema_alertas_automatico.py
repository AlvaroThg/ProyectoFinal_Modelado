"""
Sistema de Alertas Automáticas - Sabor Chapaco
Genera alertas cada noche a las 22:00 para el día siguiente
Integrado con el sistema de predicción
"""

import schedule
import time
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import json
from sistema_alertas_inteligentes import MotorAlertas, TipoAlerta, NivelAlerta

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIGURACION = {
    'hora_ejecucion': '19:58',  # Hora diaria para generar alertas
    'dias_anticipacion': 3,      # Alertas para los próximos 3 días
    'platos': ['Ranga Ranga', 'Sajta de Pollo', 'Chancho a la Cruz', 
               'Picante de Lengua', 'Chicharrón de Cerdo'],
    'inventario_simulado': {     # Inventario actual (en producción, vendrá de BD)
        'Ranga Ranga': 45,
        'Sajta de Pollo': 38,
        'Chancho a la Cruz': 30,
        'Picante de Lengua': 20,
        'Chicharrón de Cerdo': 18
    },
    'clima_pronostico': None,    # Se puede integrar con API del clima
    'eventos_calendario': {       # Eventos conocidos
        '2025-08-15': 'Festival',
        '2025-12-25': 'Feriado',
        '2026-01-01': 'Feriado'
    }
}


# ============================================================================
# CARGAR MODELO DE PREDICCIÓN
# ============================================================================

def cargar_modelo():
    """Carga el modelo de predicción"""
    try:
        model = joblib.load('modelo_final_sabor_chapaco.pkl')
        scaler = joblib.load('scaler_sabor_chapaco.pkl')
        label_encoders = joblib.load('label_encoders_sabor_chapaco.pkl')
        model_info = joblib.load('model_info.pkl')
        
        return {
            'model': model,
            'scaler': scaler,
            'encoders': label_encoders,
            'info': model_info
        }
    except Exception as e:
        print(f"⚠️ Error cargando modelo: {e}")
        return None


# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def engineer_features(data_dict):
    """Crea features para predicción"""
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


def predecir_demanda(modelo_pkg, plato, fecha, hora=19):
    """Genera predicción para un plato específico"""
    
    fecha_dt = pd.to_datetime(fecha)
    dia_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 
                  'Viernes', 'Sábado', 'Domingo'][fecha_dt.weekday()]
    
    # Obtener clima y eventos
    clima = CONFIGURACION['clima_pronostico'] or 'Soleado'
    evento = CONFIGURACION['eventos_calendario'].get(fecha.strftime('%Y-%m-%d'), 'Ninguno')
    
    input_data = {
        'fecha': fecha,
        'plato': plato,
        'hora_servicio': hora,
        'dia_semana': dia_semana,
        'condicion_climatica': clima,
        'evento_local': evento,
        'tipo_promocion': 'Ninguna',
        'es_fin_semana': 1 if fecha_dt.weekday() >= 4 else 0
    }
    
    # Feature engineering
    df_features = engineer_features(input_data)
    
    # Codificar categóricas
    for col in ['plato', 'condicion_climatica', 'evento_local', 'tipo_promocion']:
        le = modelo_pkg['encoders'][col]
        try:
            df_features[f'{col}_encoded'] = le.transform(df_features[col].astype(str))
        except ValueError:
            df_features[f'{col}_encoded'] = 0
    
    # Features finales
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
    
    X = df_features[numeric_features + categorical_encoded]
    
    # Predecir
    prediccion = modelo_pkg['model'].predict(X)[0]
    
    return max(0, prediccion), evento, clima


# ============================================================================
# GENERADOR DE ALERTAS DIARIAS
# ============================================================================

def generar_alertas_diarias():
    """
    Función principal que se ejecuta cada noche
    Genera alertas para los próximos días
    """
    
    print("\n" + "="*80)
    print(f"🚨 GENERANDO ALERTAS INTELIGENTES")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Cargar modelo
    modelo_pkg = cargar_modelo()
    if not modelo_pkg:
        print("❌ No se pudo cargar el modelo. Abortando.")
        return
    
    # Inicializar motor de alertas
    motor = MotorAlertas()
    
    # Generar alertas para los próximos N días
    dias_anticipacion = CONFIGURACION['dias_anticipacion']
    alertas_totales = []
    
    for dia in range(1, dias_anticipacion + 1):
        fecha_objetivo = datetime.now() + timedelta(days=dia)
        
        print(f"\n📅 Analizando: {fecha_objetivo.strftime('%A %d/%m/%Y')}")
        print("-" * 80)
        
        for plato in CONFIGURACION['platos']:
            
            # 1. PREDECIR DEMANDA
            prediccion, evento, clima = predecir_demanda(
                modelo_pkg, plato, fecha_objetivo
            )
            
            # 2. OBTENER INVENTARIO ACTUAL
            inventario_actual = CONFIGURACION['inventario_simulado'].get(plato, 0)
            
            # 3. CALCULAR STOCK PLANEADO (producción estimada)
            # En producción, esto vendría de la planificación
            stock_planeado = int(prediccion * 1.10)  # +10% por defecto
            
            # 4. DEMANDA PROMEDIO (últimos 7 días)
            # En producción, calcular desde BD real
            demanda_promedio = prediccion * 0.95
            
            # 5. PREDICCIONES FUTURAS (para detectar tendencias)
            predicciones_futuras = []
            for d in range(7):
                fecha_fut = fecha_objetivo + timedelta(days=d)
                pred_fut, _, _ = predecir_demanda(modelo_pkg, plato, fecha_fut)
                predicciones_futuras.append(pred_fut)
            
            # 6. GENERAR ALERTAS
            alertas = motor.generar_alertas_dia(
                prediccion=prediccion,
                fecha=fecha_objetivo,
                plato=plato,
                inventario_disponible=inventario_actual,
                stock_planeado=stock_planeado,
                condicion_clima=clima,
                tipo_evento=evento,
                demanda_promedio=demanda_promedio,
                predicciones_futuras=predicciones_futuras
            )
            
            # Mostrar alertas generadas
            if alertas:
                print(f"\n   🔔 {plato}:")
                print(f"      Predicción: {prediccion:.1f} porciones")
                print(f"      Inventario: {inventario_actual} porciones")
                print(f"      Alertas: {len(alertas)}")
                
                for alerta in alertas:
                    emoji = alerta.nivel.value['emoji']
                    print(f"      {emoji} [{alerta.nivel.name}] {alerta.tipo.value}")
                    print(f"         {alerta.mensaje}")
                
                alertas_totales.extend(alertas)
    
    # RESUMEN FINAL
    print("\n" + "="*80)
    print("📊 RESUMEN DE ALERTAS GENERADAS")
    print("="*80)
    
    alertas_criticas = motor.obtener_alertas_criticas()
    
    print(f"Total alertas: {len(alertas_totales)}")
    print(f"Alertas CRÍTICAS/ALTAS: {len(alertas_criticas)}")
    
    # Agrupar por tipo
    por_tipo = {}
    for alerta in alertas_totales:
        tipo = alerta.tipo.value
        por_tipo[tipo] = por_tipo.get(tipo, 0) + 1
    
    print("\nPor tipo:")
    for tipo, cantidad in por_tipo.items():
        print(f"   {tipo}: {cantidad}")
    
    # GUARDAR ALERTAS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'alertas_{timestamp}.json'
    
    alertas_json = {
        'fecha_generacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dias_anticipacion': dias_anticipacion,
        'total_alertas': len(alertas_totales),
        'alertas_criticas': len(alertas_criticas),
        'alertas': [a.to_dict() for a in alertas_totales]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(alertas_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Alertas guardadas en: {filename}")
    
    # ENVIAR NOTIFICACIONES (en producción)
    if alertas_criticas:
        print(f"\n🚨 ¡ATENCIÓN! {len(alertas_criticas)} alertas CRÍTICAS requieren acción inmediata:")
        for alerta in alertas_criticas[:5]:  # Mostrar las 5 más importantes
            print(f"\n   {alerta.nivel.value['emoji']} {alerta.plato} - {alerta.fecha}")
            print(f"      {alerta.mensaje}")
            print(f"      💡 {alerta.recomendacion}")
    
    print("\n✅ Proceso de alertas completado\n")
    
    return alertas_totales


# ============================================================================
# PROGRAMACIÓN AUTOMÁTICA
# ============================================================================

def iniciar_sistema_automatico():
    """
    Inicia el sistema de alertas automáticas
    Se ejecuta cada noche a la hora configurada
    """
    
    print("="*80)
    print("🤖 SISTEMA DE ALERTAS AUTOMÁTICAS - SABOR CHAPACO")
    print("="*80)
    print(f"⏰ Hora de ejecución: {CONFIGURACION['hora_ejecucion']}")
    print(f"📅 Días de anticipación: {CONFIGURACION['dias_anticipacion']}")
    print(f"🍽️  Platos monitoreados: {len(CONFIGURACION['platos'])}")
    print("="*80)
    
    # Programar ejecución diaria
    schedule.every().day.at(CONFIGURACION['hora_ejecucion']).do(generar_alertas_diarias)
    
    # TAMBIÉN ejecutar inmediatamente al inicio (para testing)
    print("\n🚀 Ejecutando primera generación de alertas...\n")
    generar_alertas_diarias()
    
    # Loop infinito esperando la hora programada
    print(f"\n⏳ Sistema activo. Esperando próxima ejecución a las {CONFIGURACION['hora_ejecucion']}...")
    print("💡 Presiona CTRL+C para detener\n")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Verificar cada minuto


# ============================================================================
# FUNCIÓN PARA EJECUTAR MANUALMENTE (TESTING)
# ============================================================================

def ejecutar_alertas_ahora():
    """Ejecuta generación de alertas inmediatamente (para testing)"""
    generar_alertas_diarias()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--now':
        # Modo manual: python sistema_alertas_automatico.py --now
        print("🔧 Modo MANUAL: Ejecutando alertas inmediatamente...\n")
        ejecutar_alertas_ahora()
    else:
        # Modo automático: python sistema_alertas_automatico.py
        print("🤖 Modo AUTOMÁTICO: Sistema programado\n")
        iniciar_sistema_automatico()