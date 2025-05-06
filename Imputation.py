import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Parameter
WINDOW = 48  # Anzahl Stunden, die als Vergleichsfenster verwendet werden

# CSV laden
df = pd.read_csv("id118/Daten/filtered_data_118.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Zeitreihe extrahieren
values = df["meter_reading"].copy()
anomalies = df["anomaly"].copy()
imputed = values.copy()

# Funktion zur Suche des ähnlichsten nicht-anomalen Fensters
def find_best_match(target_window, candidate_windows):
    best_candidate = None
    lowest_error = float("inf")
    for candidate in candidate_windows:
        mse = mean_squared_error(target_window, candidate)
        if mse < lowest_error:
            best_candidate = candidate
            lowest_error = mse
    return best_candidate

# Erzeuge eine Liste möglicher Vergleichsfenster (alle ohne Anomalien)
candidate_windows = []
for i in range(WINDOW, len(values) - WINDOW):
    window = values[i - WINDOW//2 : i + WINDOW//2]
    window_anomalies = anomalies[i - WINDOW//2 : i + WINDOW//2]
    if len(window) == WINDOW and window_anomalies.sum() == 0:
        candidate_windows.append(window.values)

# Hauptloop zur Imputation
for i in range(len(values)):
    if anomalies[i] == 1:
        start = max(0, i - WINDOW // 2)
        end = min(len(values), i + WINDOW // 2)
        target = values[start:end].values

        if len(target) == WINDOW and candidate_windows:
            best = find_best_match(target, candidate_windows)
            imputed[start:end] = best[:end - start]

# Ergebnisse speichern
df["imputed_reading"] = imputed
df.to_csv("id118/Daten/id118_imputed_output(48h).csv", index=False)

print("Imputation abgeschlossen. Datei gespeichert als 'imputed_output.csv'")
