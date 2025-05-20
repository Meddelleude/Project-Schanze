import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm  

file_path = 'lead1.0-small.csv'

output_dir = 'meter_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Lese Datei ein...")
df = pd.read_csv(file_path)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['meter_reading'] = pd.to_numeric(df['meter_reading'], errors='coerce')

print(f"Datensatzgröße: {df.shape}")
print("\nEinige Beispielzeilen:")
print(df.head())
print("\nDatentypen:")
print(df.dtypes)
print("\nFehlende Werte pro Spalte:")
print(df.isna().sum())
building_counts = df.groupby('building_id').size()
print(f"\nAnzahl der einzigartigen Building-IDs: {len(building_counts)}")
print(f"Min. Anzahl Datenpunkte pro Building: {building_counts.min()}")
print(f"Max. Anzahl Datenpunkte pro Building: {building_counts.max()}")
print(f"Durchschnittliche Anzahl Datenpunkte pro Building: {building_counts.mean():.2f}")

valid_data_counts = df.groupby('building_id')['meter_reading'].count()
completeness = valid_data_counts / building_counts

print("\nVollständigkeit der Daten (prozentual):")
print(f"Min. Vollständigkeit: {completeness.min()*100:.2f}%")
print(f"Max. Vollständigkeit: {completeness.max()*100:.2f}%")
print(f"Durchschnittliche Vollständigkeit: {completeness.mean()*100:.2f}%")

incomplete_threshold = 0.90  
incomplete_buildings = completeness[completeness < incomplete_threshold].index.tolist()

print(f"\nAnzahl der Buildings mit weniger als {incomplete_threshold*100:.0f}% gültigen Datenpunkten: {len(incomplete_buildings)}")
print(f"Prozent der Buildings mit niedriger Vollständigkeit: {len(incomplete_buildings)/len(building_counts)*100:.2f}%")
small_buildings = building_counts[building_counts < 8000].index.tolist()
print(f"\nAnzahl der Buildings mit weniger als 8000 Datenpunkten: {len(small_buildings)}")
print(f"Prozent der Buildings mit weniger als 8000 Datenpunkten: {len(small_buildings)/len(building_counts)*100:.2f}%")

buildings_to_remove = list(set(incomplete_buildings + small_buildings))
print(f"\nAnzahl der Buildings, die entfernt werden sollen (unvollständig ODER zu wenige Punkte): {len(buildings_to_remove)}")
print(f"Prozent der Buildings, die entfernt werden: {len(buildings_to_remove)/len(building_counts)*100:.2f}%")

df_filtered = df[~df['building_id'].isin(buildings_to_remove)]

print(f"\nGefilterte Datensatzgröße: {df_filtered.shape}")
print(f"Anzahl der verbleibenden Buildings: {df_filtered['building_id'].nunique()}")

filtered_file_path = 'lead1.0-filtered.csv'
df_filtered.to_csv(filtered_file_path, index=False)
print(f"\nGefilterter Datensatz gespeichert als: {filtered_file_path}")
print("\nErstelle Diagramme für jede Building-ID...")

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 12

for building_id in tqdm(df_filtered['building_id'].unique()):

    building_data = df_filtered[df_filtered['building_id'] == building_id].sort_values('timestamp')
    if building_data['meter_reading'].notna().sum() == 0:
        continue  # Überspringe dieses Building, wenn keine Messwerte vorhanden sind

    plt.figure()
    plt.plot(building_data['timestamp'], building_data['meter_reading'], 'b-', alpha=0.7)

    anomalies = building_data[building_data['anomaly'] == 1]
    if not anomalies.empty:
        plt.scatter(anomalies['timestamp'], anomalies['meter_reading'], color='red', s=50, label='Anomalien')

    plt.title(f'Smart Meter Messwerte für Building {building_id}')
    plt.xlabel('Zeitstempel')
    plt.ylabel('Messwert')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if not anomalies.empty:
        plt.legend()

    plt.savefig(os.path.join(output_dir, f'building_{building_id}.png'), dpi=100)
    plt.close()

print(f"\nAlle Diagramme wurden im Verzeichnis '{output_dir}' gespeichert.")

anomaly_count = df_filtered['anomaly'].sum()
total_readings = df_filtered.shape[0]
print(f"\nAnzahl der erkannten Anomalien: {anomaly_count}")
print(f"Prozent der Messungen mit Anomalien: {anomaly_count/total_readings*100:.4f}%")

plt.figure(figsize=(12, 6))
df_filtered['meter_reading'].hist(bins=50)
plt.title('Verteilung der Messwerte')
plt.xlabel('Messwert')
plt.ylabel('Häufigkeit')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'messwerte_verteilung.png'), dpi=100)
plt.close()

print("Analyse abgeschlossen!")