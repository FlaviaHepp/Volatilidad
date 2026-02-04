# Volatilidad
Modelo GJR-GARCH para Análisis de Volatilidad Financiera

Se buscó modelar la volatilidad condicional en retornos financieros mediante GJR-GARCH, con foco en capturar asimetrías.
Herramientas utilizadas: python, pandas, plotly, statsmodels, matplotlib, pmdarima y arch.
Estadísticas: statsmodels, arch, pmdarima para modelado y ajuste.
Visualización: seaborn, matplotlib y plotly para análisis de tendencias y estacionalidad.
Pruebas estadísticas: ADF test, Ljung-Box y análisis de correlación ACF/PACF.
**Procesos clave:**
Realicé preprocesamiento de datos financieros, garantizando estacionariedad y eliminación de ruido.
Ajusté modelos GARCH/GJR-GARCH para capturar variaciones de volatilidad durante eventos de alta incertidumbre.
Comparé modelos con métricas como AIC, BIC y pruebas de diagnóstico, seleccionando el modelo más robusto.

**Resultados:**
Predicciones precisas de volatilidad condicional, con aplicaciones en estrategias de cobertura y diversificación de riesgos.
Diseñé gráficos dinámicos y reportes interactivos, optimizando la presentación de resultados para stakeholders.

📊 Análisis de Volatilidad del Petróleo Crudo con ARIMA y GARCH

Este proyecto realiza un análisis econométrico avanzado de la volatilidad del petróleo crudo, utilizando modelos de series temporales (ARIMA) y modelos de heterocedasticidad condicional (GARCH) para capturar los principales hechos estilizados de los mercados financieros.

Los datos provienen de NYU V-Lab, cubriendo el período 1990–2023.

🎯 Objetivos del proyecto

Analizar la dinámica temporal de la volatilidad del petróleo crudo.

Explorar autocorrelación y estructura temporal de la serie.

Transformar la serie para lograr estacionariedad.

Ajustar y diagnosticar modelos ARIMA.

Modelar la volatilidad condicional mediante GARCH (1,1).

Evaluar residuos y validar supuestos estadísticos.

Realizar pronósticos con intervalos de confianza.

📁 Descripción del dataset

Fuente: NYU Volatility Labs

Activo: Petróleo crudo (Crude Oil)

Variable principal: Volatilidad diaria (vol)

Frecuencia: Diaria

Período: 01/03/1990 – 06/09/2023

Cada archivo del dataset representa la volatilidad diaria de una clase de activo o sector financiero.

🔄 Preprocesamiento de la serie

Conversión de fechas e indexado temporal.

Análisis visual de la serie original.

Primera diferenciación.

Transformación logarítmica.

Diferenciación del logaritmo para remover tendencia.

Verificación de:

Media cercana a cero

Varianza aproximadamente constante

Estas transformaciones permiten aproximar la serie a un ruido blanco estacionario.

📊 Análisis exploratorio de series temporales

Visualización de la serie original y transformada.

Gráficos de ACF y PACF para identificar estructura temporal.

Evaluación visual de autocorrelación y dependencia serial.

⏱️ Modelado ARIMA

Selección automática del modelo mediante auto_arima.

Modelo seleccionado: ARIMA(0,1,0).

Evaluación con:

AIC

Diagnósticos de residuos

Pronóstico a 100 pasos con:

Intervalos de confianza del 95 % y 80 %.

📉 Modelado de volatilidad GARCH

Modelo estimado: GARCH(1,1).

Supuestos:

Media AR

Distribución Normal

Análisis de:

Residuos estandarizados

Residuos al cuadrado

Pruebas de diagnóstico

Ljung-Box

Box-Pierce

ARCH LM Test

Los resultados validan la correcta especificación del modelo para capturar la volatilidad condicional.

🛠️ Tecnologías utilizadas

Python

pandas / numpy

Matplotlib

Plotly

statsmodels

pmdarima

arch

📂 Estructura del proyecto
├── 1.py
├── ac-vol-all-nyu/
│   └── CRUDEOIL_COM.csv
└── README.md

▶️ Cómo ejecutar el proyecto

Clonar el repositorio

git clone https://github.com/tu_usuario/nombre_del_repo.git


Instalar dependencias

pip install pandas numpy matplotlib plotly statsmodels pmdarima arch


Ejecutar el script

python 1.py

📌 Resultados principales

La serie original presenta fuerte persistencia de volatilidad.

La transformación logarítmica y diferenciación eliminan la tendencia.

El modelo ARIMA(0,1,0) describe adecuadamente la dinámica del nivel.

El modelo GARCH(1,1) captura correctamente la agrupación de volatilidad.

Los diagnósticos estadísticos validan la especificación del modelo.

📚 Aplicaciones

Gestión de riesgo financiero

Modelado de commodities

Forecasting de volatilidad

Value at Risk (VaR)

Análisis macro-financiero

⚠️ Disclaimer

Este proyecto tiene fines académicos y demostrativos.
No constituye asesoramiento financiero ni recomendaciones de inversión.

👤 Autor

Flavia Hepp
Data Science · Econometría · Finanzas Cuantitativas
