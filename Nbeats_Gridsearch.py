import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
from itertools import product

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Daten laden ---
df = pd.read_csv("filtered_data_335.csv", parse_dates=["timestamp"])

# --- Zeitreihen erstellen ---
series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
covariate_series = TimeSeries.from_dataframe(df, "timestamp", "anomaly")

# --- Skalierung ---
scaler = Scaler()
series = scaler.fit_transform(series)

# --- Split ---
train, val = series.split_after(0.8)
train_cov, val_cov = covariate_series.split_after(0.8)

# --- Early Stopping ---
early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

# --- Hyperparameter für GridSearch definieren ---
input_chunk_lengths = [168, 336, 672]  # Beispielwerte: 7 Tage, 14 Tage, 28 Tage
output_chunk_lengths = [24, 48]       # z. B. eine oder zwei Tage Vorhersage



# --- GridSearch: Alle Kombinationen der Hyperparameter durchlaufen ---
best_mae = float('inf')
best_params = {}

for input_chunk_length, output_chunk_length in product(
    input_chunk_lengths, output_chunk_lengths):
    
    # Modell initialisieren
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,

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
        verbose=False,  # Keine Ausgabe im Training
    )

    # Vorhersage erzeugen
    forecast_scaled = model.predict(n=len(val), past_covariates=covariate_series)
    forecast = scaler.inverse_transform(forecast_scaled)

    # Fehlerberechnung
    current_mae = mae(val, forecast)
    print(f"Test mit ICL={input_chunk_length}, OCL={output_chunk_length}, "
          f"-> MAE: {current_mae:.4f}")

    # Speichere die besten Hyperparameter
    if current_mae < best_mae:
        best_mae = current_mae
        best_params = {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
        }

# --- Beste Parameter ausgeben ---
print("\nBeste Hyperparameter gefunden:")
print(f"input_chunk_length: {best_params['input_chunk_length']}")
print(f"output_chunk_length: {best_params['output_chunk_length']}")
print(f"Best MAE: {best_mae:.4f}")
