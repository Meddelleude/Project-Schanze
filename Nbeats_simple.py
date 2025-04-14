import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
df = pd.read_csv("Filtered_data/filtered_data_1147.csv", parse_dates=["timestamp"])

series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
series_original = series.copy()  

scaler = Scaler()
series = scaler.fit_transform(series)

train, val = series.split_after(0.8)
val_unscaled = scaler.inverse_transform(val)

early_stopping = EarlyStopping(
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
    val_series=val,
    epochs=50,
    verbose=True
)

forecast_scaled = model.historical_forecasts(
    series,
    start=0.8,                    
    forecast_horizon=1,
    stride=1,
    retrain=False,
    verbose=True
)

forecast = scaler.inverse_transform(forecast_scaled)

mae_val = mae(val_unscaled, forecast)
mean_val = val_unscaled.values().mean()
mae_percent = (mae_val / mean_val) * 100
rmse_val = rmse(val_unscaled, forecast)

print(f"MAE: {mae_val:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"Prozentualer MAE: {mae_percent:.2f}%")

building_id = int(df["building_id"].iloc[0])
fig, ax = plt.subplots(figsize=(16, 8))
series_original.plot(label="Echte Werte", ax=ax, linewidth=0.5)
forecast.plot(label="Vorhersage", ax=ax, linewidth=0.5)
ax.legend()
ax.set_xlabel("Zeit", fontsize=12)
ax.set_ylabel("Messwerte", fontsize=12)
plt.title(f"forecast_plot_simple_id{building_id},{timestamp}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, MAE%: {mae_percent:.2f}, ICL: {model.input_chunk_length}, OCL: {model.output_chunk_length}")
plt.savefig(f"Forecast/forecast_plot_simple_id{building_id},{timestamp}.png")
plt.show()

forecast_df = forecast.pd_dataframe()
forecast_df.to_csv(f"Forecast/forecast_data_id{building_id},{timestamp}.csv", index=True)
