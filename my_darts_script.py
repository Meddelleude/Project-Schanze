import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
import pytorch_lightning as pl

# Daten laden
df = pd.read_csv("data_set_1.csv", parse_dates=["timestamp"])

# Zeitreihen-Objekt erstellen
series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
series2 = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
# Skalierung anwenden
scaler = Scaler()
series = scaler.fit_transform(series)

# Trainings- und Testdaten aufteilen
train, val = series.split_after(0.8)

# Early Stopping konfigurieren
early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, mode="min"
)

# NBEATS-Modell initialisieren
model = NBEATSModel(
    input_chunk_length=48,
    output_chunk_length=24,
    random_state=42,
    pl_trainer_kwargs={"callbacks": [early_stopping]},
)

# Modell trainieren
model.fit(train, val_series=val, epochs=10, verbose=True)

# Vorhersage erzeugen
forecast = model.predict(n=1680)
forecast = scaler.inverse_transform(forecast)  # Zurückskalieren

# Plots erstellen
fig, ax = plt.subplots()
series2.plot(label="Echte Werte", ax=ax)
forecast.plot(label="Vorhersage", ax=ax)
ax.legend()
plt.savefig("forecast_plot.png")  # Speichert den Plot
plt.show()

# Vorhersage als CSV speichern
forecast_df = forecast.pd_dataframe()
forecast_df.to_csv("forecast_data.csv", index=True)

# Fehler berechnen
print("MAE:", mae(val, model.predict(len(val))))
print("RMSE:", rmse(val, model.predict(len(val))))