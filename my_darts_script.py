import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Daten laden ---
df = pd.read_csv("data_set_1.csv", parse_dates=["timestamp"])

# --- Zeitreihen erstellen ---
series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
series_original = series.copy()  # für späteren Vergleich
covariate_series = TimeSeries.from_dataframe(df, "timestamp", "anomaly")

# --- Skalierung der Zielvariable ---
scaler = Scaler()
series = scaler.fit_transform(series)

# --- Split ---
train, val = series.split_after(0.8)
train_cov, val_cov = covariate_series.split_after(0.8)

# --- Modell mit EarlyStopping ---
early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

model = NBEATSModel(
    input_chunk_length=336,
    output_chunk_length=48,
    random_state=42,
    pl_trainer_kwargs={"callbacks": [early_stopping]},
)

# --- Training ---
model.fit(
    series=train,
    past_covariates=train_cov,
    val_series=val,
    val_past_covariates=val_cov,
    epochs=50,
    verbose=True,
)

# --- Realistische Vorhersage mit historical_forecasts ---
forecast_scaled = model.historical_forecasts(
    series,
    past_covariates=covariate_series,
    start=0.8,              # ab Validierungsbereich
    forecast_horizon=1,
    stride=1,
    retrain=False,
    verbose=True
)

# --- Zurückskalieren (Forecast + Ground Truth) ---
forecast = scaler.inverse_transform(forecast_scaled)
val_unscaled = scaler.inverse_transform(val)

# --- Fehlerberechnung ---
mae_val = mae(val_unscaled, forecast)
rmse_val = rmse(val_unscaled, forecast)
mean_val = val_unscaled.values().mean()
mae_percent = (mae_val / mean_val) * 100

# --- Ergebnisse anzeigen ---
print(f"MAE: {mae_val:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"Prozentualer MAE: {mae_percent:.2f}%")

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 8))
series_original.plot(label="Echte Werte", ax=ax, linewidth=0.5)
forecast.plot(label="Vorhersage", ax=ax, linewidth=0.5)
ax.legend()
ax.set_xlabel("Zeit", fontsize=12)
ax.set_ylabel("Messwerte", fontsize=12)
plt.title(f"forecast_plot_{timestamp}, MAE: {mae_val:.4f}, MAE%: {mae_percent:.2f}%, RMSE: {rmse_val:.4f}")
plt.savefig(f"Forecast/forecast_plot_{timestamp}.png")
plt.show()

# --- Forecast als CSV speichern ---
forecast_df = forecast.pd_dataframe()
forecast_df.to_csv(f"Forecast/forecast_data_{timestamp}.csv", index=True)
