import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
from datetime import datetime, timedelta

raw_data = pd.read_csv('lead1.0-small.csv')
raw_data_copy = raw_data.copy()
raw_data_copy.dropna(inplace=True)
raw_data_copy['timestamp'] = pd.to_datetime(raw_data_copy['timestamp'])

def count_building_ids_dict(df):
    
    if 'building_id' not in df.columns:
        raise ValueError("Die Spalte 'building_id' existiert nicht in der Datei.")
    
    id_counts = df['building_id'].value_counts()
    return id_counts

def write_as_file(df):
    with open("count.txt", "w") as file:
        for building_id, count in df.items(): 
            file.write(f"{building_id} {count}\n")

def filter_most_common_id(raw_data_copy, id_counts):
    for i in range(6):
        target_id = id_counts.index[i]
        
        filtered_df = raw_data_copy[raw_data_copy['building_id'] == target_id]

        filtered_df.to_csv(f"filtered_data_{i+1}.txt", sep=" ", index=False)
    
    return filtered_df

def build_plot(raw_data_copy, id_counts):
    for i in range(6):
        target_id = id_counts.index[i]
        
        filtered_df = raw_data_copy[raw_data_copy['building_id'] == target_id]
        plt.figure(figsize=(10, 6)) 
        plt.plot(filtered_df['timestamp'], filtered_df['meter_reading'], label='Meter Reading', color='blue')

        plt.title(f'Meter Reading Over Time {target_id}')  
        plt.xlabel('Timestamp')  
        plt.ylabel('Meter Reading')  
        plt.xticks(rotation=45)  
        plt.grid(True)  

        plt.savefig(f'meter_reading_plot{i+1}.png', bbox_inches='tight')

def filter_id_out(raw_data_copy, id):
    filtered_df = raw_data_copy[raw_data_copy['building_id'] == id]
    filtered_df.to_csv(f"Filtered_data/filtered_data_{id}.csv", sep=",", index=False)
    df = pd.read_csv(f"Filtered_data/filtered_data_{id}.csv")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    start = df.index.min()
    end = df.index.max()
    voller_index = pd.date_range(start=start, end=end, freq='H')

    df = df.reindex(voller_index)

    print("Fehlende Werte nach Reindexing:\n", df.isna().sum())

    df_interpoliert = df.interpolate(method='time')
    df_interpoliert.index.name = 'timestamp'

    df_interpoliert.to_csv(f"Filtered_data/filtered_data_{id}.csv")

def run(df):
    id_counts = df['building_id'].value_counts()
    for i in range(6):
        target_id = id_counts.index[i]
        
        filtered_df = raw_data_copy[raw_data_copy['building_id'] == target_id]

        filtered_df.to_csv(f"filtered_data_{i+1}.txt", sep=" ", index=False)
        plt.figure(figsize=(15, 6))  
        sns.lineplot(x=filtered_df['timestamp'], y=filtered_df['meter_reading'], label='Meter Reading', color='blue', linewidth=0.5)

        plt.title(f'Meter Reading Over Time {target_id}')  
        plt.xlabel('Timestamp')  
        plt.ylabel('Meter Reading')  
        plt.xticks(rotation=45)  
        plt.grid(True)  

        plt.savefig(f'meter_reading_plot_{target_id}.png', bbox_inches='tight') 

def run_all_in_one(df):

    id_counts = df['building_id'].value_counts()
    palette = sns.color_palette("Set2", n_colors=6)  
    plt.figure(figsize=(30, 6))  

    for i in range(6):
        target_id = id_counts.index[i]  
        filtered_df = df[df['building_id'] == target_id]
        sns.lineplot(x=filtered_df['timestamp'], y=filtered_df['meter_reading'], label=f'ID {target_id}', color=palette[i], linewidth=0.5)

    plt.title('Meter Reading Over Time for Different Building IDs')  
    plt.xlabel('Timestamp')  
    plt.ylabel('Meter Reading')  
    plt.xticks(rotation=45)  
    plt.grid(True)  

    plt.legend(title="Building ID", loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig('combined_meter_reading_plot.png', bbox_inches='tight')
    plt.show()
    
def plot_one_id(df, id):
    
    plt.figure(figsize=(30, 6))
    filtered_df = df[df['building_id'] == id]
    sns.lineplot(x=filtered_df['timestamp'], y=filtered_df['meter_reading'], label=f'ID {id}', linewidth=0.5)
    plt.title('Meter Reading Over Time for Different Building IDs')  
    plt.xlabel('Timestamp')  
    plt.ylabel('Meter Reading')  
    plt.xticks(rotation=45)  
    plt.grid(True)  

    plt.legend(title="Building ID", loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(f'Filtered_data/meter_reading_plot_{id}.png', bbox_inches='tight')
    plt.show()

def stunden_im_jahr():
    jahr= int(input("Bitte gibt das Jahr ein: "))
    if(jahr%4==0 & jahr%100!=0):
        stunden = 24*366
    elif(jahr%400==0):
         stunden = 24*366
    else:
        stunden = 24*365
    print(stunden)

def anomaly(df):
    anomaly_counts = df.groupby('building_id')['anomaly'].value_counts().unstack(fill_value=0)
    anomaly_counts.to_csv("anomaly_counts.txt", sep=" ")
    anomaly_counts_sorted = anomaly_counts.sort_values(by=1, ascending=False)
    anomaly_counts_sorted.to_csv("anomaly_counts_sorted.txt", sep=" ")
    print(anomaly_counts_sorted)

def count_building_id_and_anomalies(df):
    id_counts = df['building_id'].value_counts()
    anomaly_counts = df[df['anomaly'] == 1].groupby('building_id').size()
    
    combined_counts = pd.DataFrame({
        'building_id_count': id_counts,
        'anomaly_count': anomaly_counts
    })

    combined_counts_sorted = combined_counts.sort_values(by='building_id_count', ascending=False)
    combined_counts_sorted.to_csv("combined_counts.txt", sep=" ")
    return combined_counts

def only_anomaly(input_file):
    # CSV-Datei einlesen
    df = pd.read_csv(input_file, parse_dates=["timestamp"])
    
    # Nur Anomalien filtern
    df = df[df["anomaly"] == 1].sort_values("timestamp")

    # Sicherstellen, dass Anomalien vorhanden sind
    if df.empty:
        print("Keine Anomalien gefunden.")
        return

    # Zeitdifferenz in Stunden zwischen aufeinanderfolgenden Anomalien berechnen
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().div(3600).fillna(0)

    # Neues Cluster starten, wenn die Zeitdifferenz > 1h ist
    df["cluster_id"] = (df["time_diff"] > 1).cumsum()

    # Datei mit Leerzeilen zwischen Clustern vorbereiten
    output_rows = []
    for _, group in df.groupby("cluster_id"):
        output_rows.append(group.drop(columns=["time_diff", "cluster_id"]))
        # Leere Zeile einfügen
        empty_row = pd.DataFrame([[""] * len(df.columns[:-2])], columns=df.columns[:-2])
        output_rows.append(empty_row)

    # Alles zusammenführen
    output_df = pd.concat(output_rows, ignore_index=True)

    # Datei schreiben
    building_id = df["building_id"].iloc[0]
    output_file = f"anomalies_building_{building_id}.txt"
    output_df.to_csv(output_file, sep=" ", index=False)

    print(f"Clustered Anomalien gespeichert in: {output_file}")

def to_csv(input):
    df = pd.read_csv(input, sep=" ", engine="python")
    df.to_csv("data_set_1.csv", index=False)

def anomalie_ersetzen():
    df = pd.read_csv("Filtered_data/filtered_data_335.csv", parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    df.loc[df["anomaly"] == 1, "meter_reading"] = pd.NA

    df["meter_reading"] = df["meter_reading"].interpolate(method='time', limit_direction='both')

    df["anomaly"] = 0

    df.to_csv("id335/anomalien_entfernt.csv")

def anomalie_anteile_erstellen():
    df_original  = pd.read_csv("id121/Daten/filtered_data_121.csv", parse_dates=["timestamp"])
    df_original.set_index("timestamp", inplace=True)

    anomalie_indices = df_original[df_original["anomaly"] == 1].index

    for prozent in range(10, 101, 10):
        df = df_original.copy()

        n = int(len(anomalie_indices) * (prozent / 100))

        ersatz_indices = np.random.choice(anomalie_indices, size=n, replace=False)

        df.loc[ersatz_indices, "meter_reading"] = pd.NA

        df["meter_reading"] = df["meter_reading"].interpolate(method="time", limit_direction="both")

        df.loc[ersatz_indices, "anomaly"] = 0

        df.to_csv(f"id121/Daten/anomalien_ersetzt_{prozent}prozent.csv")
        print(f"{prozent}% ersetzt – Datei gespeichert: anomalien_ersetzt_{prozent}prozent.csv")

def remove_anomaly_clusters(input_file, output_folder, cluster_gap_hours=2, modus='rückwärts'):

    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(input_file, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True) 

    def find_anomaly_clusters(df, gap_threshold='2h'):
        anomalies = df[df['anomaly'] == 1].copy()
        anomalies = anomalies.sort_index()
        anomalies['time_diff'] = anomalies.index.to_series().diff()
        anomalies['cluster_id'] = (anomalies['time_diff'] > pd.Timedelta(gap_threshold)).cumsum()
        return anomalies

    current_df = df.copy()
    iteration = 0

    while True:
        anomalies = find_anomaly_clusters(current_df, gap_threshold=f'{cluster_gap_hours}h')

        if anomalies.empty:
            print("Keine weiteren Anomalien mehr vorhanden.")
            break

        if modus == 'vorwärts':
            cluster_id = anomalies['cluster_id'].iloc[0] 
        elif modus == 'rückwärts':
            cluster_id = anomalies['cluster_id'].iloc[-1] 
        elif modus == 'zufällig':
            cluster_id = random.choice(anomalies['cluster_id'].unique())  
        else:
            raise ValueError("Ungültiger Modus. Bitte 'vorwärts', 'rückwärts' oder 'zufällig' wählen.")

        cluster_indices = anomalies[anomalies['cluster_id'] == cluster_id].index

        if cluster_indices.empty:
            break

        current_df.loc[cluster_indices, 'meter_reading'] = np.nan
        current_df.loc[cluster_indices, 'anomaly'] = 0  
        current_df['meter_reading'] = current_df['meter_reading'].interpolate(method='time')
        output_file = os.path.join(output_folder, f'anomalien_ersetzt_{(iteration+1)*10}prozent.csv')
        current_df.reset_index().to_csv(output_file, index=False)  
        print(f"{(iteration+1)*10}% der Cluster ersetzt – Datei gespeichert: {output_file}")

        iteration += 1

def plot_verschiedene_ids_zusammen(df, ids):
    os.makedirs("Filtered_data", exist_ok=True)
    
    palette = sns.color_palette("Set2", n_colors=len(ids))  # Farbschema dynamisch nach Anzahl IDs
    plt.figure(figsize=(30, 6))  # Großer Plot

    for i, target_id in enumerate(ids):
        filtered_df = df[df['building_id'] == target_id]
        
        if filtered_df.empty:
            print(f"⚠️ Keine Daten für Gebäude-ID {target_id}. Überspringe...")
            continue
        
        sns.lineplot(x=filtered_df['timestamp'], y=filtered_df['meter_reading'], 
                     label=f'ID {target_id}', color=palette[i % len(palette)], linewidth=0.5)

    plt.title('Meter Reading Over Time for Selected Building IDs')  
    plt.xlabel('Timestamp')  
    plt.ylabel('Meter Reading')  
    plt.xticks(rotation=45)  
    plt.grid(True)  

    plt.legend(title="Building ID", loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig('combined_meter_reading_plot2.png', bbox_inches='tight')
    plt.show()
        
def fill_missing_timesteps(file_path, building_id, output_dir=None,
                          time_column='timestamp', value_column='meter_reading', 
                          anomaly_column='anomaly', time_freq='1h'):
    print(f"Extrahiere und vervollständige Daten für Gebäude-ID: {building_id}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Datei geladen: {file_path}")
        print(f"Gesamtzahl der Datenpunkte: {len(df)}")
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")
        return None

    building_df = df[df['building_id'] == building_id].copy()
    
    if len(building_df) == 0:
        print(f"Keine Daten für Gebäude-ID {building_id} gefunden.")
        return None
    print(f"Datenpunkte für Gebäude-ID {building_id}: {len(building_df)}")
    if building_df[time_column].dtype != 'datetime64[ns]':
        building_df[time_column] = pd.to_datetime(building_df[time_column])
    building_df = building_df.sort_values(by=time_column)
    
    if df[time_column].dtype != 'datetime64[ns]':
        df[time_column] = pd.to_datetime(df[time_column])
    
    min_date = df[time_column].min()
    max_date = df[time_column].max()
    
    print(f"Vollständiger Zeitbereich des Datensatzes: {min_date} bis {max_date}")

    complete_timerange = pd.date_range(start=min_date, end=max_date, freq=time_freq)
    expected_count = len(complete_timerange)
    print(f"Erwartete Anzahl an Datenpunkten für ein volles Jahr: {expected_count}")

    existing_timestamps = set(building_df[time_column])
    missing_timestamps = [ts for ts in complete_timerange if ts not in existing_timestamps]
    
    print(f"Anzahl fehlender Zeitstempel: {len(missing_timestamps)}")
    if len(missing_timestamps) > 0:
        print(f"Erste 5 fehlende Zeitstempel: {missing_timestamps[:5]}")

    complete_df = pd.DataFrame({time_column: complete_timerange})
    building_df_indexed = building_df.set_index(time_column)
    complete_df_indexed = complete_df.set_index(time_column)
    merged_df = pd.merge(complete_df_indexed, building_df_indexed, 
                         left_index=True, right_index=True, 
                         how='left')

    if 'building_id' in merged_df.columns:
        merged_df['building_id'] = merged_df['building_id'].fillna(building_id)
    else:
        merged_df['building_id'] = building_id
    if value_column in merged_df.columns:
        missing_count = merged_df[value_column].isna().sum()
        print(f"Anzahl fehlender Messwerte: {missing_count}")
        merged_df[value_column] = merged_df[value_column].fillna(0)
    else:
        merged_df[value_column] = 0
    
    if anomaly_column in merged_df.columns:
        merged_df[anomaly_column] = merged_df[anomaly_column].fillna(0)
    else:
        merged_df[anomaly_column] = 0
    merged_df = merged_df.reset_index()
    merged_df = merged_df.sort_values(by=time_column)
    original_count = len(building_df)
    filled_count = len(merged_df) - original_count
    
    print(f"Ursprüngliche Datenpunkte: {original_count}")
    print(f"Hinzugefügte Datenpunkte: {filled_count}")
    print(f"Gesamtzahl der Datenpunkte nach Auffüllung: {len(merged_df)}")

    if len(merged_df) != expected_count:
        print(f"WARNUNG: Anzahl der Datenpunkte nach Auffüllung ({len(merged_df)}) "
              f"entspricht nicht der erwarteten Anzahl ({expected_count})!")

        merged_timestamps = set(merged_df[time_column])
        still_missing = [ts for ts in complete_timerange if pd.Timestamp(ts) not in merged_timestamps]
        if still_missing:
            print(f"Es fehlen immer noch {len(still_missing)} Zeitstempel!")
            print(f"Erste 5 fehlende: {still_missing[:5]}")
        
        duplicate_timestamps = merged_df[time_column].duplicated().sum()
        if duplicate_timestamps > 0:
            print(f"Es gibt {duplicate_timestamps} doppelte Zeitstempel!")

    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    
    if output_dir is None:
        output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"{file_base}_building_{building_id}_filled.csv")
    
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"Ergebnis gespeichert in: {output_file}")
        return output_file
    except Exception as e:
        print(f"Fehler beim Speichern der Datei: {e}")
        return None


#run(raw_data_copy)
#run_all_in_one(raw_data_copy)
#anomaly(raw_data_copy)
#count_building_id_and_anomalies(raw_data_copy)
only_anomaly("Filtered_data/filtered_data_254.csv")
#to_csv("filtered_data_1.txt")
#stunden_im_jahr()
#plot_one_id(raw_data_copy, 685)
#plot_verschiedene_ids_zusammen(raw_data_copy,ids = [118, 335, 439])
#filter_id_out(raw_data_copy, 254)
#anomalie_ersetzen()
#anomalie_anteile_erstellen()
#remove_anomaly_clusters('id118/Daten/filtered_data_118.csv', 'id118/Daten4', modus='zufällig')
#fill_missing_timesteps("lead1.0-small.csv",254)