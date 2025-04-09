import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    filtered_df.to_csv(f"filtered_data_{id}.csv", sep=",", index=False)
    df = pd.read_csv(f"filtered_data_{id}.csv")

    # 2. Zeitstempel parsen und als Index setzen
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 3. Vollständigen Zeitindex erzeugen (für ein ganzes Jahr, stündlich)
    start = df.index.min()
    end = df.index.max()
    voller_index = pd.date_range(start=start, end=end, freq='H')

    # 4. DataFrame auf vollständigen Zeitindex ausrichten
    df = df.reindex(voller_index)

    # 5. Fehlende Werte anzeigen (sollte jetzt die 3 NaNs zeigen)
    print("Fehlende Werte nach Reindexing:\n", df.isna().sum())

    # 6. Interpolation der fehlenden Werte (zeitlich)
    df_interpoliert = df.interpolate(method='time')
    df_interpoliert.index.name = 'timestamp'
    # 7. Optional: Ergebnis speichern
    df_interpoliert.to_csv(f"filtered_data_{id}.csv")

def run(df):
    id_counts = df['building_id'].value_counts()
    for i in range(6):
        target_id = id_counts.index[i]
        
        filtered_df = raw_data_copy[raw_data_copy['building_id'] == target_id]

        filtered_df.to_csv(f"filtered_data_{i+1}.txt", sep=" ", index=False)
        plt.figure(figsize=(15, 6))  
        sns.lineplot(x=filtered_df['timestamp'], y=filtered_df['meter_reading'], label='Meter Reading', color='blue', linewidth=0.5)

        # Titel und Achsenbeschriftungen
        plt.title(f'Meter Reading Over Time {target_id}')  
        plt.xlabel('Timestamp')  
        plt.ylabel('Meter Reading')  
        plt.xticks(rotation=45)  # Drehen der X-Achsen-Beschriftungen
        plt.grid(True)  

        # Speichern des Plots
        plt.savefig(f'meter_reading_plot_{target_id}.png', bbox_inches='tight') 

def run_all_in_one(df):
    # Zähle die Häufigkeit der building_id
    id_counts = df['building_id'].value_counts()

    # Wähle eine Farbpalette aus Seaborn
    palette = sns.color_palette("Set2", n_colors=6)  # Hier wählst du eine Farbpalette aus

    plt.figure(figsize=(30, 6))  # Erstelle das große Diagramm für alle Linien

    # Schleife über die ersten 6 häufigsten building_ids
    for i in range(6):
        target_id = id_counts.index[i]  # Hol dir die building_id des aktuellen Rangs

        # Filtere die Daten für diese building_id
        filtered_df = df[df['building_id'] == target_id]

        # Plot der Linie für diese building_id mit einer einzigartigen Farbe aus der Palette
        sns.lineplot(x=filtered_df['timestamp'], y=filtered_df['meter_reading'], label=f'ID {target_id}', color=palette[i], linewidth=0.5)

    # Titel und Achsenbeschriftungen
    plt.title('Meter Reading Over Time for Different Building IDs')  
    plt.xlabel('Timestamp')  
    plt.ylabel('Meter Reading')  
    plt.xticks(rotation=45)  # Drehen der X-Achsen-Beschriftungen
    plt.grid(True)  

    # Füge eine Legende hinzu
    plt.legend(title="Building ID", loc="upper left", bbox_to_anchor=(1, 1))

    # Speichern des kombinierten Plots
    plt.savefig('combined_meter_reading_plot.png', bbox_inches='tight')

    # Zeige den Plot an
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
    # Zählen der building_id Vorkommen
    id_counts = df['building_id'].value_counts()
    
    # Zählen der Anomalien für jede building_id
    anomaly_counts = df[df['anomaly'] == 1].groupby('building_id').size()
    
    # Kombinieren der beiden Zählungen in einem DataFrame
    combined_counts = pd.DataFrame({
        'building_id_count': id_counts,
        'anomaly_count': anomaly_counts
    })
    
    # Sortieren nach 'building_id_count' in absteigender Reihenfolge
    combined_counts_sorted = combined_counts.sort_values(by='building_id_count', ascending=False)
    combined_counts_sorted.to_csv("combined_counts.txt", sep=" ")
    return combined_counts

def only_anomaly(input_file):
    
    df = pd.read_csv(input_file, sep=" ", engine="python")  
    df_filtered = df[df['anomaly'] == 1] 
    building_id = df_filtered['building_id'].iloc[0]
    output_file = f"anomalies_building_{building_id}.txt"
    df_filtered.to_csv(output_file, sep=" ", index=False)

def to_csv(input):
    df = pd.read_csv(input, sep=" ", engine="python")
    df.to_csv("data_set_1.csv", index=False)

#run(raw_data_copy)
#run_all_in_one(raw_data_copy)
#anomaly(raw_data_copy)
#count_building_id_and_anomalies(raw_data_copy)
#only_anomaly("filtered_data_6.txt")
#to_csv("filtered_data_1.txt")
#stunden_im_jahr()
filter_id_out(raw_data_copy, 335)