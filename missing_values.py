import pandas as pd

# === Parameter ===
csv_path = 'lead1.0-small.csv'
output_path = 'mit_missing_markierung_building_950.csv'
building_id_to_process = 950  # <<< Hier gewünschte ID eintragen

# === Daten einlesen ===
df = pd.read_csv(csv_path, parse_dates=['timestamp'])
df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)

# === Daten für gewünschte Building-ID filtern ===
df_building = df[df['building_id'] == building_id_to_process].copy()
if df_building.empty:
    raise ValueError(f"Keine Daten gefunden für building_id = {building_id_to_process}")

# === Vollständige Zeitreihe für diese ID erzeugen ===
start = df_building['timestamp'].min().floor('H')
end = df_building['timestamp'].max().ceil('H')
full_range = pd.date_range(start=start, end=end, freq='H')
df_full = pd.DataFrame({'timestamp': full_range})
df_full['building_id'] = building_id_to_process

# === Mergen ===
df_merged = pd.merge(df_full, df_building, on=['timestamp', 'building_id'], how='left')

# === Fehlende Werte markieren ===
missing_mask = df_merged['meter_reading'].isna()
missing_count = missing_mask.sum()

# Fehlende Werte setzen
df_merged.loc[missing_mask, 'meter_reading'] = 0.0
df_merged.loc[missing_mask, 'anomaly'] = 1

# Nicht-fehlende Anomalien beibehalten oder 0 setzen
df_merged['anomaly'] = df_merged['anomaly'].fillna(0)

# === Ergebnis speichern ===
df_merged.to_csv(output_path, index=False)

# === Ausgabe in Konsole ===
print(f"Building ID: {building_id_to_process}")
print(f"Zeitraum: {start} bis {end}")
print(f"Anzahl fehlender Zeitpunkte erkannt und ersetzt: {missing_count}")
print(f"Erweiterter Datensatz gespeichert unter: {output_path}")
