import pandas as pd
from darts import TimeSeries


df = pd.read_csv("data_set_1.csv", parse_dates=["timestamp"])
series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
from darts.dataprocessing.transformers import Scaler
scaler = Scaler()
series = scaler.fit_transform(series)
series.plot()

from darts.models import NBEATSModel

# Modell initialisieren
model = NBEATSModel(
    input_chunk_length=48,  # Vergangene Werte, die als Input genutzt werden
    output_chunk_length=24,  # Wie viele Werte vorhergesagt werden
    random_state=42
)

# Trainings- und Testdaten aufteilen
train, val = series.split_after(0.8)

# Modell trainieren
model.fit(train, epochs=50, verbose=True)