import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
import pytorch_lightning as pl
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Daten laden
df = pd.read_csv("data_set_1.csv", parse_dates=["timestamp"])

# Zeitreihen-Objekt erstellen
series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
series2 = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
covariate_series = TimeSeries.from_dataframe(df, "timestamp", "anomaly")

# Skalierung anwenden
scaler = Scaler()
#covariate_scaled = scaler.fit_transform(covariate_series)
covariate_scaled = covariate_series
series = scaler.fit_transform(series)

# Trainings- und Testdaten aufteilen
train_cov, val_cov = covariate_series.split_after(0.8)
train, val = series.split_after(0.8)

# Early Stopping konfigurieren
early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, mode="min"
)

# NBEATS-Modell initialisieren
model = NBEATSModel(
    input_chunk_length=400,
    output_chunk_length=80,
    random_state=42,
    pl_trainer_kwargs={"callbacks": [early_stopping]},
)

# Modell trainieren
model.fit(
    series=train,
    past_covariates=train_cov,
    val_series=val,
    val_past_covariates=val_cov,
    epochs=50,
    verbose=True
)

# Vorhersage erzeugen
forecast = model.predict(n=len(val), past_covariates=covariate_scaled)
forecast = scaler.inverse_transform(forecast)
Mae = mae(val, model.predict(len(val),past_covariates=covariate_scaled))
RMSE = rmse(val, model.predict(len(val),past_covariates=covariate_scaled))

# Plots erstellen
fig, ax = plt.subplots(figsize=(16, 8))
series2.plot(label="Echte Werte", ax=ax,linewidth=0.5)
forecast.plot(label="Vorhersage", ax=ax,linewidth=0.5)
ax.legend()
ax.set_xlabel("Zeit", fontsize=12)
ax.set_ylabel("Messwerte", fontsize=12)
plt.legend()
plt.title(f"forecast_plot_{timestamp},Mae:  {Mae}, RMSE: {RMSE},ICL: {model.input_chunk_length}, OCL: {model.output_chunk_length} ")
plt.savefig(f"Forecast/forecast_plot_{timestamp}.png")  # Speichert den Plot
plt.show()

# Vorhersage als CSV speichern
forecast_df = forecast.pd_dataframe()
forecast_df.to_csv(f"Forecast/forecast_data_{timestamp}.csv", index=True)

# Fehler berechnen
print("MAE:", mae(val, model.predict(len(val),past_covariates=covariate_scaled)))
print("RMSE:", rmse(val, model.predict(len(val),past_covariates=covariate_scaled)))
