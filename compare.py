import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, mape
from darts.dataprocessing.transformers import Scaler
import pytorch_lightning as pl
from datetime import datetime

def train_and_evaluate(df, use_anomaly=False):
    series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)
    train, val = series_scaled.split_after(0.8)

    if use_anomaly:
        covariate = TimeSeries.from_dataframe(df, "timestamp", "anomaly")
        train_cov, val_cov = covariate.split_after(0.8)
    else:
        train_cov = val_cov = None

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    model = NBEATSModel(
        input_chunk_length=336,
        output_chunk_length=48,
        random_state=42,
        pl_trainer_kwargs={"callbacks": [early_stopping]},
    )

    model.fit(
        series=train,
        past_covariates=train_cov,
        val_series=val,
        val_past_covariates=val_cov,
        epochs=50,
        verbose=False
    )

    full_cov = covariate if use_anomaly else None
    forecast = model.predict(n=len(val), past_covariates=full_cov)
    forecast = scaler.inverse_transform(forecast)
    val_actual = scaler.inverse_transform(val)

    mae_score = mae(val_actual, forecast)
    mape_score = mape(val_actual, forecast)
    mean_val_scalar = val_actual.univariate_values().mean()
    mae_percent = (mae_score / mean_val_scalar) * 100

    return {
        "mae": mae_score,
        "mae_percent": mae_percent,
        "mape": mape_score,
        "label": "mit Anomalie" if use_anomaly else "ohne Anomalie"
    }

# Lade deine Datei
df = pd.read_csv("Filtered_data/filtered_data_335.csv", parse_dates=["timestamp"])
building_id = df["building_id"].iloc[0] if "building_id" in df.columns else "Unbekannt"

results = [
    train_and_evaluate(df, use_anomaly=False),
    train_and_evaluate(df, use_anomaly=True),
]

vergleich_text = f"Modellvergleich für Gebäude {building_id}:\n\n"
for res in results:
    vergleich_text += (
        f"{res['label']}:\n"
        f"- MAE:          {res['mae']:.4f}\n"
        f"- MAE_prozent:      {res['mae_percent']:.2f}%\n"
        f"- MAPE:         {res['mape']:.2f}%\n\n"
    )

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"Modellvergleich_Gebaeude_{building_id}_{timestamp}.txt"
with open(filename, "w") as f:
    f.write(vergleich_text)

print(f"Ergebnisse gespeichert in: {filename}")
