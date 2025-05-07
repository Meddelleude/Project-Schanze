import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("id118/Daten/filtered_data_118.csv", index_col=0, parse_dates=True)
df_result = pd.read_csv("imputed_data_dd_cpi.csv", index_col=0, parse_dates=True)
def validate_dd_cpi(original_df: pd.DataFrame, imputed_df: pd.DataFrame) -> dict:
    """
    Validiert die DD-CPI-Imputation anhand mehrerer Kriterien:
    1. Energieerhaltung im Gap
    2. Durchschnittlicher Fehler auf normalen Tagen
    3. Vergleich des Tagesverbrauchs vor und nach der Imputation
    4. Visualisierung des Energieverlaufs an einem Beispieltag mit Anomalie

    Parameter:
    - original_df: DataFrame mit den Originaldaten, enthält Spalten 'meter_reading' und 'anomaly'
    - imputed_df: DataFrame mit den imputierten Daten, enthält Spalte 'meter_reading_imputed'

    Rückgabe:
    - Ein Dictionary mit den Validierungsergebnissen
    """

    results = {}

    # Sicherstellen, dass die Indizes datetime-Objekte sind
    original_df = original_df.copy()
    imputed_df = imputed_df.copy()
    original_df.index = pd.to_datetime(original_df.index)
    imputed_df.index = pd.to_datetime(imputed_df.index)

    # Datum extrahieren
    original_df["date"] = original_df.index.date
    imputed_df["date"] = imputed_df.index.date

    # Beispieltag mit Anomalie
    anomalous_days = original_df[original_df["anomaly"] == 1]["date"].unique()
    if len(anomalous_days) == 0:
        raise ValueError("Keine Anomalien in den Originaldaten gefunden.")
    example_day = anomalous_days[0]

    # 1. Energieerhaltung im Gap
    def gap_energy(df, day):
        day_data = df[df["date"] == day]
        if day_data.empty:
            return None
        before = day_data.iloc[0]["meter_reading_imputed"]
        after = day_data.iloc[-1]["meter_reading_imputed"]
        return round(after - before, 2)

    results["Beispieltag mit Anomalie"] = str(example_day)
    results["Energie im Gap (Beispieltag)"] = gap_energy(imputed_df, example_day)

    # 2. Durchschnittlicher Fehler auf normalen Tagen
    normal_mask = original_df["anomaly"] == 0
    if "meter_reading_imputed" not in imputed_df.columns:
        raise ValueError("Die Spalte 'meter_reading_imputed' fehlt im imputierten DataFrame.")
    diff = imputed_df.loc[normal_mask, "meter_reading_imputed"] - original_df.loc[normal_mask, "meter_reading"]
    results["Ø Fehler auf normalen Tagen (kWh)"] = round(diff.abs().mean(), 4)

    # 3. Tagesverbrauch vor und nach der Imputation
    daily_before = original_df["meter_reading"].resample("D").agg(lambda x: x.max() - x.min())
    daily_after = imputed_df["meter_reading_imputed"].resample("D").agg(lambda x: x.max() - x.min())
    delta = (daily_after - daily_before).abs()
    results["Max. Differenz Tagesverbrauch (kWh)"] = round(delta.max(), 2)
    results["Ø Differenz Tagesverbrauch (kWh)"] = round(delta.mean(), 2)

    # 4. Visualisierung des Energieverlaufs am Beispieltag
    df_orig_example = original_df[original_df["date"] == example_day]
    df_imp_example = imputed_df[imputed_df["date"] == example_day]

    plt.figure(figsize=(10, 4))
    plt.plot(df_orig_example.index, df_orig_example["meter_reading"], label="Original (mit Anomalie)", linestyle="--")
    plt.plot(df_imp_example.index, df_imp_example["meter_reading_imputed"], label="Nach DD-CPI", linestyle="-")
    plt.title(f"Energieverlauf am Beispieltag {example_day}")
    plt.ylabel("Energie [kWh]")
    plt.xlabel("Zeit")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
results = validate_dd_cpi(df, df_result)
print(results)