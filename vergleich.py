import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime

bericht = ""
building_id = None  # wird beim ersten Durchlauf gesetzt

for prozent in range(10, 101, 10):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Datei laden
    file_path = f"id118/Daten4/anomalien_ersetzt_{prozent}prozent.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    if building_id is None:
        building_id = int(df["building_id"].iloc[0])

    # Zeitreihe vorbereiten
    series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
    scaler = Scaler()
    series = scaler.fit_transform(series)
    train, val = series.split_after(0.8)
    val_unscaled = scaler.inverse_transform(val)

    # EarlyStopping & Modell
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    model = NBEATSModel(
        input_chunk_length=336,
        output_chunk_length=48,
        random_state=42,
        pl_trainer_kwargs={"callbacks": [early_stopping]},
    )

    model.fit(series=train, val_series=val, epochs=50, verbose=False)

    # Vorhersage
    forecast_scaled = model.historical_forecasts(
        series,
        start=0.8,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=False
    )
    forecast = scaler.inverse_transform(forecast_scaled)

    # Metriken
    mae_val = mae(val_unscaled, forecast)
    mean_val = val_unscaled.values().mean()
    mae_percent = (mae_val / mean_val) * 100
    rmse_val = rmse(val_unscaled, forecast)

    # Text für Bericht
    bericht += f"{prozent}% ersetzte Anomalien:\n"
    bericht += f"- MAE:          {mae_val:.4f}\n"
    bericht += f"- RMSE:         {rmse_val:.4f}\n"
    bericht += f"- MAE (%):      {mae_percent:.2f} %\n\n"

# Bericht speichern
gesamtbericht = f"Modellvergleich für Gebäude {building_id}\n" + "="*40 + "\n\n" + bericht
filename = f"Forecast/Modellvergleich_daten4_Building_{building_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

with open(filename, "w") as f:
    f.write(gesamtbericht)

print(f"Bericht gespeichert unter: {filename}")
