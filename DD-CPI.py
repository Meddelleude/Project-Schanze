import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("id118/Daten/filtered_data_118.csv", parse_dates=["timestamp"])
# ==== Parameter ====
column_power = "meter_reading"     # Eingabespalte: Power (Verbrauch pro Stunde)
column_anomaly = "anomaly"
seasonal_cycle_days = 365
weights = (5, 1, 10)  # (we, ww, ws)

# ==== Hilfsfunktionen ====

def calculate_energy_series(power_series):
    return power_series.cumsum()

def weekday_class(day):
    return 0 if day.weekday() < 5 else 1

def seasonal_pos(dt):
    return dt.dayofyear

def calculate_dissimilarity(day_i, day_j, Ei, Ej, we, ww, ws):
    De = abs(Ei - Ej) / (max(Ei, Ej) + 1e-6)
    Dw = 0.0 if day_i.weekday() == day_j.weekday() else (
        0.5 if weekday_class(day_i) == weekday_class(day_j) else 1.0
    )
    s = seasonal_cycle_days
    si, sj = seasonal_pos(day_i), seasonal_pos(day_j)
    Ds = min(abs(si - sj), s - abs(si - sj)) / (s // 2)
    return we * De + ww * Dw + ws * Ds

# ==== DD-CPI Imputation ====

def dd_cpi_impute(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    df["power"] = df[column_power].astype(float)
    df["energy"] = calculate_energy_series(df["power"])  # kumulierte Energie
    df["date"] = df.index.date

    # Tagesverbrauch berechnen
    daily_energy = df.groupby("date")["energy"].agg(["first", "last"])
    daily_energy["consumption"] = daily_energy["last"] - daily_energy["first"]

    # Prophet-Muster
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(daily_energy.index),
        "y": daily_energy["consumption"].fillna(method='ffill')
    })
    m = Prophet(weekly_seasonality=True, daily_seasonality=False)
    m.fit(prophet_df)
    forecast = m.predict(prophet_df[["ds"]])
    daily_energy["pattern"] = forecast["weekly"].values

    anomaly_days = df[df[column_anomaly] == 1]["date"].unique()
    complete_days = df[df[column_anomaly] == 0]["date"].unique()

    for day in anomaly_days:
        try:
            gap_mask = df["date"] == day
            gap_idx = df[gap_mask].index

            # Gap-Grenzen bestimmen
            gap_start = gap_idx[0] - timedelta(hours=1)
            gap_end = gap_idx[-1] + timedelta(hours=1)

            if gap_start not in df.index or gap_end not in df.index:
                continue

            Ei = df.loc[gap_end, "energy"] - df.loc[gap_start, "energy"]
            if Ei <= 0:
                continue  # unplausibel

            best_score = float("inf")
            best_day = None

            for ref_day in complete_days:
                ref_day_str = str(ref_day)
                if ref_day_str not in daily_energy.index:
                    continue
                Ej = daily_energy.loc[ref_day_str, "consumption"]
                score = calculate_dissimilarity(
                    pd.to_datetime(day), pd.to_datetime(ref_day), Ei, Ej, *weights
                )
                if score < best_score:
                    best_score = score
                    best_day = ref_day

            # Imputation
            template = df[df["date"] == best_day]["power"].values
            if len(template) != len(gap_idx):
                continue

            Ej_hat = np.sum(template)
            if Ej_hat <= 0:
                continue

            scale = Ei / Ej_hat
            df.loc[gap_mask, "power"] = template * scale

        except Exception as e:
            print(f"Fehler bei Tag {day}: {e}")
            continue

    # Ergebnis als neue Spalte
    df["meter_reading_imputed"] = df["power"]
    return df.reset_index()

df_result = dd_cpi_impute(df)

# Speichern
df_result.to_csv("imputed_data_dd_cpi.csv")