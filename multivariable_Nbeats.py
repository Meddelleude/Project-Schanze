import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

# CSV laden
df = pd.read_csv("data_set_1.csv", parse_dates=["timestamp"])

# Optional: Daten sortieren und Duplikate entfernen
df = df.sort_values("timestamp").drop_duplicates("timestamp")

# Optional: Nur Zeilen mit vollständigen Daten
df = df.dropna(subset=["meter_reading", "anomaly"])

# Metering-Zeitreihe
series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")

# Anomaly-Zeitreihe als Covariate
covariate_series = TimeSeries.from_dataframe(df, "timestamp", "anomaly")

# Beide synchron? Check!
print("Startzeitpunkt series:", series.start_time())
print("Startzeitpunkt anomaly:", covariate_series.start_time())
print("Länge series:", len(series))
print("Länge anomaly:", len(covariate_series))
print(df["anomaly"].value_counts())
df[df["anomaly"] == 1].head()