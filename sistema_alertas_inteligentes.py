"""
Sistema de Alertas Inteligentes - Sabor Chapaco
Genera alertas basadas en predicciones de demanda y simulación
Examen Final - Modelado y Simulación de Sistemas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import json


# ============================================================================
# TIPOS DE ALERTAS
# ============================================================================

class TipoAlerta(Enum):
    """Tipos de alertas del sistema"""
    SOBREVENTA = "sobreventa"
    MERMA_ALTA = "merma_alta"
    CLIMA = "clima"
    TURISMO = "turismo"
    STOCK_BAJO = "stock_bajo"
    COMPRA_URGENTE = "compra_urgente"
    TENDENCIA_ALCISTA = "tendencia_alcista"
    EVENTO_ESPECIAL = "evento_especial"


class NivelAlerta(Enum):
    """Niveles de severidad de alertas"""
    CRITICO = {"nivel": 4, "color": "#8F0B13", "emoji": "🔴"}
    ALTO = {"nivel": 3, "color": "#FF6B35", "emoji": "🟠"}
    MEDIO = {"nivel": 2, "color": "#FFB800", "emoji": "🟡"}
    BAJO = {"nivel": 1, "color": "#4CAF50", "emoji": "🟢"}
    INFO = {"nivel": 0, "color": "#2196F3", "emoji": "ℹ️"}


# ============================================================================
# CLASE ALERTA
# ============================================================================

class Alerta:
    """Representa una alerta del sistema"""
    
    def __init__(self, tipo, nivel, mensaje, fecha, plato=None, 
                 valor_actual=None, valor_esperado=None, recomendacion=None):
        self.tipo = tipo
        self.nivel = nivel
        self.mensaje = mensaje
        self.fecha = fecha
        self.plato = plato
        self.valor_actual = valor_actual
        self.valor_esperado = valor_esperado
        self.recomendacion = recomendacion
        self.timestamp = datetime.now()
    
    def to_dict(self):
        """Convierte la alerta a diccionario"""
        return {
            'tipo': self.tipo.value,
            'nivel': self.nivel.name,
            'nivel_emoji': self.nivel.value['emoji'],
            'mensaje': self.mensaje,
            'fecha': self.fecha.strftime('%Y-%m-%d') if isinstance(self.fecha, datetime) else str(self.fecha),
            'plato': self.plato,
            'valor_actual': self.valor_actual,
            'valor_esperado': self.valor_esperado,
            'recomendacion': self.recomendacion,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def __str__(self):
        emoji = self.nivel.value['emoji']
        return f"{emoji} [{self.nivel.name}] {self.tipo.value.upper()}: {self.mensaje}"


# ============================================================================
# MOTOR DE ALERTAS
# ============================================================================

class MotorAlertas:
    """
    Motor principal del sistema de alertas inteligentes
    Genera alertas basadas en predicciones y configuración
    """
    
    def __init__(self, configuracion=None):
        self.configuracion = configuracion or self._configuracion_default()
        self.alertas_generadas = []
        self.historial_alertas = []
    
    def _configuracion_default(self):
        """Configuración por defecto del sistema de alertas"""
        return {
            'sobreventa': {
                'habilitado': True,
                'umbral_porcentaje': 0.90,  # Alerta si demanda > 90% inventario
                'dias_anticipacion': 1
            },
            'merma': {
                'habilitado': True,
                'umbral_porcentaje': 0.30,  # Alerta si desperdicio > 30%
                'costo_porcion': 15  # Bs por porción desperdiciada
            },
            'clima': {
                'habilitado': True,
                'incremento_lluvia': 0.15,  # +15% en platos calientes
                'incremento_frio': 0.20     # +20% en platos calientes
            },
            'turismo': {
                'habilitado': True,
                'meses_temporada': [5, 6, 7, 8, 9, 10],  # Mayo-Octubre
                'incremento_demanda': 0.28  # +28% en temporada
            },
            'stock': {
                'habilitado': True,
                'stock_minimo_dias': 1.5,  # Mínimo stock para 1.5 días
                'margen_seguridad': 0.20   # +20% sobre predicción
            }
        }
    
    def actualizar_configuracion(self, nueva_config):
        """Actualiza configuración del sistema"""
        self.configuracion.update(nueva_config)
        print("✅ Configuración actualizada")
    
    
    # ========================================================================
    # ALERTA 1: SOBREVENTA
    # ========================================================================
    
    def evaluar_sobreventa(self, prediccion, inventario_disponible, fecha, plato):
        """
        Evalúa riesgo de sobreventa (agotamiento de stock)
        Alerta cuando: demanda_predicha > inventario_disponible
        """
        
        if not self.configuracion['sobreventa']['habilitado']:
            return None
        
        umbral = self.configuracion['sobreventa']['umbral_porcentaje']
        ratio_demanda_inventario = prediccion / inventario_disponible if inventario_disponible > 0 else np.inf
        
        # CRÍTICO: Demanda muy superior a inventario
        if ratio_demanda_inventario > 1.5:
            alerta = Alerta(
                tipo=TipoAlerta.SOBREVENTA,
                nivel=NivelAlerta.CRITICO,
                mensaje=f"⚠️ RIESGO CRÍTICO DE SOBREVENTA para {plato}",
                fecha=fecha,
                plato=plato,
                valor_actual=inventario_disponible,
                valor_esperado=prediccion,
                recomendacion=f"URGENTE: Aumentar producción en {int(prediccion - inventario_disponible)} porciones. "
                              f"Considerar activar producción extra o compra de emergencia."
            )
            return alerta
        
        # ALTO: Demanda cerca del límite
        elif ratio_demanda_inventario >= umbral:
            faltante = int(prediccion - inventario_disponible)
            alerta = Alerta(
                tipo=TipoAlerta.SOBREVENTA,
                nivel=NivelAlerta.ALTO,
                mensaje=f"Riesgo de agotamiento: {plato} - Faltan {faltante} porciones",
                fecha=fecha,
                plato=plato,
                valor_actual=inventario_disponible,
                valor_esperado=prediccion,
                recomendacion=f"Incrementar producción en {faltante + int(prediccion * 0.1)} porciones "
                              f"(incluye margen de seguridad del 10%)"
            )
            return alerta
        
        return None
    
    
    # ========================================================================
    # ALERTA 2: MERMA ALTA
    # ========================================================================
    
    def evaluar_merma(self, prediccion, stock_planeado, fecha, plato):
        """
        Evalúa riesgo de merma (desperdicio excesivo)
        Alerta cuando: stock_planeado >> demanda_predicha
        """
        
        if not self.configuracion['merma']['habilitado']:
            return None
        
        umbral = self.configuracion['merma']['umbral_porcentaje']
        costo_porcion = self.configuracion['merma']['costo_porcion']
        
        desperdicio_estimado = stock_planeado - prediccion
        porcentaje_merma = desperdicio_estimado / stock_planeado if stock_planeado > 0 else 0
        
        # ALTO: Merma significativa
        if desperdicio_estimado > 10 and porcentaje_merma >= umbral:
            costo_perdida = desperdicio_estimado * costo_porcion
            
            alerta = Alerta(
                tipo=TipoAlerta.MERMA_ALTA,
                nivel=NivelAlerta.ALTO if porcentaje_merma >= 0.50 else NivelAlerta.MEDIO,
                mensaje=f"Alto riesgo de merma en {plato}: {int(desperdicio_estimado)} porciones",
                fecha=fecha,
                plato=plato,
                valor_actual=stock_planeado,
                valor_esperado=prediccion,
                recomendacion=f"Reducir producción a {int(prediccion * 1.15)} porciones (incluye 15% margen). "
                              f"Pérdida estimada evitada: Bs. {costo_perdida:.2f}. "
                              f"Considerar promoción 2x1 si ya se produjo exceso."
            )
            return alerta
        
        return None
    
    
    # ========================================================================
    # ALERTA 3: CLIMA
    # ========================================================================
    
    def evaluar_clima(self, prediccion_base, condicion_clima, fecha, plato, tipo_plato="caliente"):
        """
        Ajusta predicción según condiciones climáticas
        Días lluviosos/fríos → +15-20% demanda en platos calientes
        """
        
        if not self.configuracion['clima']['habilitado']:
            return None
        
        # Categorías de platos sensibles al clima
        platos_calientes = ['Ranga Ranga', 'Sajta de Pollo', 'Picante de Lengua']
        
        if plato not in platos_calientes:
            return None
        
        incremento = 0
        nivel = NivelAlerta.INFO
        mensaje_clima = ""
        
        if condicion_clima.lower() in ['lluvioso', 'lluvia']:
            incremento = self.configuracion['clima']['incremento_lluvia']
            nivel = NivelAlerta.MEDIO
            mensaje_clima = "día lluvioso"
        elif condicion_clima.lower() in ['frio', 'frío', 'nublado_frio']:
            incremento = self.configuracion['clima']['incremento_frio']
            nivel = NivelAlerta.MEDIO
            mensaje_clima = "día frío"
        
        if incremento > 0:
            nueva_prediccion = prediccion_base * (1 + incremento)
            incremento_porciones = nueva_prediccion - prediccion_base
            
            alerta = Alerta(
                tipo=TipoAlerta.CLIMA,
                nivel=nivel,
                mensaje=f"Ajuste por clima: {plato} - Se espera {mensaje_clima}",
                fecha=fecha,
                plato=plato,
                valor_actual=prediccion_base,
                valor_esperado=nueva_prediccion,
                recomendacion=f"Incrementar producción en {int(incremento_porciones)} porciones "
                              f"({incremento*100:.0f}% más por condiciones climáticas). "
                              f"Nueva meta: {int(nueva_prediccion)} porciones."
            )
            return alerta
        
        return None
    
    
    # ========================================================================
    # ALERTA 4: TEMPORADA TURÍSTICA
    # ========================================================================
    
    def evaluar_temporada_turistica(self, prediccion_base, fecha, plato):
        """
        Ajusta predicción en temporada turística
        Mayo-Octubre: +28% demanda general
        """
        
        if not self.configuracion['turismo']['habilitado']:
            return None
        
        mes_actual = fecha.month if isinstance(fecha, datetime) else pd.to_datetime(fecha).month
        meses_temporada = self.configuracion['turismo']['meses_temporada']
        incremento = self.configuracion['turismo']['incremento_demanda']
        
        # Verificar si estamos en temporada turística
        if mes_actual in meses_temporada:
            nueva_prediccion = prediccion_base * (1 + incremento)
            incremento_porciones = nueva_prediccion - prediccion_base
            
            # Detectar inicio de temporada (primer mes)
            if mes_actual == meses_temporada[0]:
                nivel = NivelAlerta.ALTO
                mensaje_extra = " - ¡INICIO DE TEMPORADA!"
            else:
                nivel = NivelAlerta.INFO
                mensaje_extra = ""
            
            alerta = Alerta(
                tipo=TipoAlerta.TURISMO,
                nivel=nivel,
                mensaje=f"Temporada turística activa{mensaje_extra}",
                fecha=fecha,
                plato=plato,
                valor_actual=prediccion_base,
                valor_esperado=nueva_prediccion,
                recomendacion=f"Incrementar stock en {int(incremento_porciones)} porciones "
                              f"({incremento*100:.0f}% más por turismo). "
                              f"Meta ajustada: {int(nueva_prediccion)} porciones. "
                              f"Considerar contratar personal temporal."
            )
            return alerta
        
        return None
    
    
    # ========================================================================
    # ALERTA 5: STOCK BAJO
    # ========================================================================
    
    def evaluar_stock_minimo(self, inventario_actual, demanda_diaria_promedio, plato, fecha):
        """
        Verifica si el stock está por debajo del mínimo recomendado
        Mínimo = demanda_promedio * días_mínimos + margen_seguridad
        """
        
        if not self.configuracion['stock']['habilitado']:
            return None
        
        dias_minimos = self.configuracion['stock']['stock_minimo_dias']
        margen = self.configuracion['stock']['margen_seguridad']
        
        stock_minimo_recomendado = demanda_diaria_promedio * dias_minimos * (1 + margen)
        
        if inventario_actual < stock_minimo_recomendado:
            deficit = stock_minimo_recomendado - inventario_actual
            
            # CRÍTICO: Stock para menos de 1 día
            if inventario_actual < demanda_diaria_promedio:
                nivel = NivelAlerta.CRITICO
                mensaje = f"⚠️ STOCK CRÍTICO: {plato} - Solo quedan {int(inventario_actual)} porciones"
            # ALTO: Stock insuficiente
            else:
                nivel = NivelAlerta.ALTO
                mensaje = f"Stock bajo para {plato}"
            
            alerta = Alerta(
                tipo=TipoAlerta.STOCK_BAJO,
                nivel=nivel,
                mensaje=mensaje,
                fecha=fecha,
                plato=plato,
                valor_actual=inventario_actual,
                valor_esperado=stock_minimo_recomendado,
                recomendacion=f"COMPRA URGENTE: Reabastecer {int(deficit)} porciones inmediatamente. "
                              f"Stock objetivo: {int(stock_minimo_recomendado)} porciones "
                              f"(cubre {dias_minimos} días + {margen*100:.0f}% margen)"
            )
            return alerta
        
        return None
    
    
    # ========================================================================
    # ALERTA 6: TENDENCIA ALCISTA
    # ========================================================================
    
    def evaluar_tendencia(self, predicciones_serie, plato, fecha):
        """
        Detecta tendencias alcistas significativas en la demanda
        Útil para planificación a mediano plazo
        """
        
        if len(predicciones_serie) < 7:
            return None
        
        # Calcular tendencia (regresión lineal simple)
        x = np.arange(len(predicciones_serie))
        y = np.array(predicciones_serie)
        
        # Pendiente de la recta de regresión
        pendiente = np.polyfit(x, y, 1)[0]
        
        # Porcentaje de cambio
        cambio_porcentual = (pendiente * len(predicciones_serie)) / y.mean()
        
        # TENDENCIA ALCISTA SIGNIFICATIVA
        if cambio_porcentual > 0.15:  # +15% en el periodo
            alerta = Alerta(
                tipo=TipoAlerta.TENDENCIA_ALCISTA,
                nivel=NivelAlerta.INFO,
                mensaje=f"📈 Tendencia alcista detectada en {plato}",
                fecha=fecha,
                plato=plato,
                valor_actual=y[0],
                valor_esperado=y[-1],
                recomendacion=f"Demanda creciendo {cambio_porcentual*100:.1f}% en los próximos {len(predicciones_serie)} días. "
                              f"Considerar: aumentar capacidad de producción, negociar mejores precios con proveedores, "
                              f"evaluar aumentar precio si demanda supera oferta."
            )
            return alerta
        
        return None
    
    
    # ========================================================================
    # ALERTA 7: EVENTO ESPECIAL
    # ========================================================================
    
    def evaluar_evento_especial(self, fecha, tipo_evento, prediccion_base, plato):
        """
        Ajusta predicción para eventos especiales
        Festivales, feriados, eventos locales
        """
        
        incrementos_eventos = {
            'Festival': 0.31,      # +31% demanda
            'Fiesta': 0.25,        # +25% demanda
            'Feriado': 0.18        # +18% demanda
        }
        
        if tipo_evento in incrementos_eventos:
            incremento = incrementos_eventos[tipo_evento]
            nueva_prediccion = prediccion_base * (1 + incremento)
            incremento_porciones = nueva_prediccion - prediccion_base
            
            alerta = Alerta(
                tipo=TipoAlerta.EVENTO_ESPECIAL,
                nivel=NivelAlerta.ALTO if tipo_evento == 'Festival' else NivelAlerta.MEDIO,
                mensaje=f"🎉 Evento especial: {tipo_evento} - Ajuste de demanda para {plato}",
                fecha=fecha,
                plato=plato,
                valor_actual=prediccion_base,
                valor_esperado=nueva_prediccion,
                recomendacion=f"Preparar {int(incremento_porciones)} porciones extra "
                              f"({incremento*100:.0f}% más por {tipo_evento}). "
                              f"Meta: {int(nueva_prediccion)} porciones. "
                              f"Considerar reforzar personal y verificar disponibilidad de ingredientes."
            )
            return alerta
        
        return None
    
    
    # ========================================================================
    # PROCESAMIENTO MASIVO DE ALERTAS
    # ========================================================================
    
    def generar_alertas_dia(self, prediccion, fecha, plato, 
                           inventario_disponible=None,
                           stock_planeado=None,
                           condicion_clima=None,
                           tipo_evento=None,
                           demanda_promedio=None,
                           predicciones_futuras=None):
        """
        Genera todas las alertas aplicables para un día específico
        """
        
        alertas_dia = []
        
        # 1. Sobreventa
        if inventario_disponible is not None:
            alerta = self.evaluar_sobreventa(prediccion, inventario_disponible, fecha, plato)
            if alerta:
                alertas_dia.append(alerta)
        
        # 2. Merma
        if stock_planeado is not None:
            alerta = self.evaluar_merma(prediccion, stock_planeado, fecha, plato)
            if alerta:
                alertas_dia.append(alerta)
        
        # 3. Clima
        if condicion_clima:
            alerta = self.evaluar_clima(prediccion, condicion_clima, fecha, plato)
            if alerta:
                alertas_dia.append(alerta)
        
        # 4. Turismo
        alerta = self.evaluar_temporada_turistica(prediccion, fecha, plato)
        if alerta:
            alertas_dia.append(alerta)
        
        # 5. Stock bajo
        if inventario_disponible is not None and demanda_promedio is not None:
            alerta = self.evaluar_stock_minimo(inventario_disponible, demanda_promedio, plato, fecha)
            if alerta:
                alertas_dia.append(alerta)
        
        # 6. Tendencia
        if predicciones_futuras and len(predicciones_futuras) >= 7:
            alerta = self.evaluar_tendencia(predicciones_futuras, plato, fecha)
            if alerta:
                alertas_dia.append(alerta)
        
        # 7. Evento especial
        if tipo_evento:
            alerta = self.evaluar_evento_especial(fecha, tipo_evento, prediccion, plato)
            if alerta:
                alertas_dia.append(alerta)
        
        self.alertas_generadas.extend(alertas_dia)
        return alertas_dia
    
    
    def obtener_alertas_criticas(self):
        """Retorna solo alertas críticas y altas"""
        return [a for a in self.alertas_generadas 
                if a.nivel in [NivelAlerta.CRITICO, NivelAlerta.ALTO]]
    
    
    def resumen_alertas(self):
        """Genera resumen estadístico de alertas"""
        if not self.alertas_generadas:
            return "No hay alertas generadas"
        
        df_alertas = pd.DataFrame([a.to_dict() for a in self.alertas_generadas])
        
        resumen = {
            'total_alertas': len(self.alertas_generadas),
            'por_nivel': df_alertas['nivel'].value_counts().to_dict(),
            'por_tipo': df_alertas['tipo'].value_counts().to_dict(),
            'alertas_criticas': len(self.obtener_alertas_criticas())
        }
        
        return resumen
    
    
    def exportar_alertas_json(self, ruta='alertas_sabor_chapaco.json'):
        """Exporta alertas a JSON"""
        alertas_dict = [a.to_dict() for a in self.alertas_generadas]
        
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(alertas_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Alertas exportadas a {ruta}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("🚨 SISTEMA DE ALERTAS INTELIGENTES - SABOR CHAPACO")
    print("="*80)
    
    # Crear motor de alertas
    motor = MotorAlertas()
    
    # Configuración personalizada (opcional)
    motor.actualizar_configuracion({
        'sobreventa': {'umbral_porcentaje': 0.85},  # Más sensible
        'merma': {'costo_porcion': 18}  # Costo ajustado
    })
    
    # Simulación de predicción para un día
    print("\n📊 Generando alertas para: Ranga Ranga - 2025-08-15\n")
    
    alertas = motor.generar_alertas_dia(
        prediccion=55,
        fecha=datetime(2025, 8, 15),
        plato='Ranga Ranga',
        inventario_disponible=45,      # Stock actual bajo
        stock_planeado=70,             # Stock planeado excesivo
        condicion_clima='Lluvioso',    # Clima favorable
        tipo_evento='Festival',        # Evento especial
        demanda_promedio=48,
        predicciones_futuras=[48, 50, 52, 54, 56, 58, 60]  # Tendencia alcista
    )
    
    # Mostrar alertas
    print(f"🔔 {len(alertas)} alertas generadas:\n")
    for alerta in alertas:
        print(alerta)
        print(f"   💡 Recomendación: {alerta.recomendacion}\n")
    
    # Resumen
    print("="*80)
    print("📊 RESUMEN DE ALERTAS")
    print("="*80)
    resumen = motor.resumen_alertas()
    print(json.dumps(resumen, indent=2))
    
    # Exportar
    motor.exportar_alertas_json()
    
    print("\n✨ Sistema de alertas funcionando correctamente")