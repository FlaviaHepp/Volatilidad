
import numpy as np 
import pandas as pd 
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import os


crude_oil = pd.read_csv('CRUDEOIL_COM.csv').set_index('date')
crude_oil.head()

fig = px.line(crude_oil, x = crude_oil.index, y = 'vol', title='Precios diarios del petróleo crudo\n', template = "plotly_dark")
fig.show()

fig, ax = plt.subplots(2, 1, figsize = (15,15))
sm.graphics.tsa.plot_acf(crude_oil, title = 'Autocorrelación del precio del petróleo crudo\n', lags = 100, ax = ax[0])
sm.graphics.tsa.plot_pacf(crude_oil, title = 'Autocorrelación parcial del precio del petróleo crudo\n', lags = 50, ax = ax[1])
plt.show()

#Dado que el fondo ACF está decayendo lentamente, indica la presencia de tendencias en los datos. Eliminemos la tendencia mediante el método de diferenciación. A continuación se muestra el 
#primer gráfico de diferenciación.

# Gráfico de la primera diferenciación del precio del petróleo crudo

crude_oil_first_diff = crude_oil.diff().dropna()
fig = px.line(crude_oil_first_diff, x = crude_oil_first_diff.index, y = 'vol', title = 'Primera diferencia del precio del petróleo crudo\n', template = "plotly_dark")
fig.show()

#still we can see the volatility after doing the first differencing of the dataset. Let's try the difference of log of crude oil prices.

crude_oil_log = np.log(crude_oil)
fig = px.line(crude_oil_log, x = crude_oil_log.index, y = 'vol', title = 'Registro del precio del petróleo crudo\n', template = "plotly_dark")
fig.show()

#Log de los precios del petróleo crudo son mucho más suaves que los de los precios de las materias primas.

# Gráfico de primera diferenciación del logaritmo del precio del petróleo crudo
crude_oil_log_first_diff = np.log(crude_oil).diff().dropna()
fig = px.line(crude_oil_log_first_diff, x = crude_oil_log_first_diff.index, y = 'vol', title = 'Registro de primera diferencia del precio del petróleo crudo\n', template = "plotly_dark")
fig.show()

#En general, los datos ahora parecen tener una variación más estable que antes. Se puede considerar la diferencia logarítmica para eliminar la 
# tendencia del conjunto de datos. Ahora revisemos nuevamente los gráficos de acf y pacf para ver la diferencia de logaritmos de los precios del petróleo crudo.

fig, ax = plt.subplots(2, 1, figsize = (15,15))
sm.graphics.tsa.plot_acf(crude_oil_log_first_diff, title = 'Diferencia del logaritmo del precio del petróleo crudo: autocorrelación\n', lags = 50, ax = ax[0])
sm.graphics.tsa.plot_pacf(crude_oil_log_first_diff, title = 'Diferencia del logaritmo del precio del petróleo crudo: autocorrelación parcial\n', lags = 50, ax = ax[1])
plt.show()

#A partir de los gráficos ACF y PACF anteriores de la diferencia del logaritmo de los precios del petróleo crudo, se observa que no existe una correlación serial presente para la diferencia del logaritmo de los precios del petróleo crudo. Ahora, después de la diferencia logarítmica, los datos podrían haberse convertido en series temporales de ruido blanco. Comprobemos algunas estadísticas básicas para estos datos.

crude_oil_log_first_diff.mean()

crude_oil_log_first_diff.std()

#En el gráfico anterior hemos observado que la varianza es estable para la diferencia del logaritmo de los precios del petróleo crudo, y además 
# la media es casi cero. Estas dos observaciones son suficientes para afirmar que los datos ahora se han convertido en datos de series temporales 
# de ruido blanco. Dado que los datos son ruido blanco normal, los datos no siguen ningún patrón. Sin embargo, hagamos un análisis más detallado 
# construyendo modelos ARIMA.

# Selección eficiente del modelo ARIMA
mod_can_auto = auto_arima(
    np.log(crude_oil).dropna(),  # stepwise=False,
    start_p=0,
    start_d=0,
    start_q=0,
    max_p=3,
    max_d=3,
    max_q=3,
    trace=True,
    with_intercept=False,
    return_valid_fits=True,
)

#El mejor modelo se parece a ARIMA(0,1,0) ya que tiene el valor AIC más bajo

# Mejor modelo ARIMA para el precio del petróleo crudo
mod_can_a = ARIMA(
    np.log(crude_oil), order=(0, 1, 0), trend="n"
).fit()  # Este es el mejor modelo en implementación de Python.
print(mod_can_a.summary())

# Informe de diagnóstico para el modelo ARIMA(0,1,0)
mod_can_a.plot_diagnostics()
plt.show()

# Gráfico de pronóstico de ARIMA (0,1,0) con un intervalo de confianza del 95%
plt.rcParams["figure.figsize"] = (16, 9)
# Trazar datos de Google
# fig,ax = plt.figure(figsize = (15,5))
ffx = crude_oil.copy()
ffx.index = [i for i in range(len(ffx))]  # Establecer índice numérico
ffx.plot(ylabel="Precio del petróleo crudo", title="Pronóstico de ARIMA(0,1,0)\n", color="m")

# obtener datos de pronóstico para los próximos 100 pasos
forecast = mod_can_a.get_forecast(steps=100)
forecast_mean = np.exp(forecast.predicted_mean)  # media de los datos de pronóstico
conf_int95 = forecast.conf_int(alpha=0.05)  # intervalo de confianza del 95%
conf_int80 = forecast.conf_int(alpha=0.2)  # intervalo de confianza del 80%

# Trazar el pronóstico medio y los intervalos de confianza del 95% y 80%.
plt.plot(forecast_mean, c="b")
plt.fill_between(
    conf_int95.index,
    np.exp(conf_int95["lower vol"]),
    np.exp(conf_int95["upper vol"]),
    color="b",
    alpha=0.3,
)
plt.fill_between(
    conf_int80.index,
    np.exp(conf_int80["lower vol"]),
    np.exp(conf_int80["upper vol"]),
    color="b",
    alpha=0.5,
)
plt.grid()
plt.show()

np.exp(conf_int80["lower vol"]).min()

np.exp(conf_int95["upper vol"]).max()

#GARCH
fig, ax = plt.subplots(2, 1, figsize = (15,15))
sm.graphics.tsa.plot_acf(crude_oil_log, title = 'Registro del precio del petróleo crudo: autocorrelación\n', lags = 50, ax = ax[0])
sm.graphics.tsa.plot_pacf(crude_oil_log, title = 'Registro del precio del petróleo crudo: autocorrelación parcial\n', lags = 50, ax = ax[1])
plt.show()

# GARCH(1,1) Modelo con ruido blanco normal
garch11_spec = arch_model(
    crude_oil_log.dropna(),
    vol="GARCH",
    p=1,
    q=1,
    mean="AR",
    dist="Normal",
    rescale=True,
)
garch11_fit = garch11_spec.fit()
garch11_fit.summary()

#Estimador de covarianza: robusto
garch11_fit.std_resid

# Pruebas de diagnóstico para el modelo GARCH(1,1) con ruido blanco normal
print("GARCH(1,1) Modelo con ruido blanco normal\n")

# Prueba de Ljung-Box y prueba de Box-Pierce
print("Pruebas Ljung-Box y Box-Pierce sobre residuos estandarizados")
print(acorr_ljungbox(garch11_fit.std_resid, boxpierce=True))

print("\nPruebas de Ljung-Box y Box-Pierce sobre residuos cuadrados estandarizados")
print(acorr_ljungbox(garch11_fit.std_resid**2, boxpierce=True))

# Prueba ARCH LM para heterocedasticidad condicional
print("\nPrueba ARCH LM para heterocedasticidad condicional")
print(garch11_fit.arch_lm_test(standardized=True))

