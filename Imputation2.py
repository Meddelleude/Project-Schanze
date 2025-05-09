import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Daten einlesen
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Umwandlung des Zeitstempels in ein datetime-Objekt
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Setze timestamp als Index für einfachere zeitbasierte Operationen
    df.set_index('timestamp', inplace=True)
    return df

# Funktion zur Extraktion von Zeitmerkmalen
def extract_time_features(df):
    df_features = df.copy()
    # Zeitliche Merkmale hinzufügen
    df_features['hour'] = df_features.index.hour
    df_features['day'] = df_features.index.day
    df_features['month'] = df_features.index.month
    df_features['weekday'] = df_features.index.weekday
    return df_features

# DD-CPI Implementierung
def dd_cpi_imputation(df, k=5):
    """
    Implementierung der Data-Driven Copy-Paste Imputation
    
    Parameter:
    - df: DataFrame mit timestamp als Index, meter_reading und anomaly Spalten
    - k: Anzahl der ähnlichsten Muster, die für die Imputation verwendet werden sollen
    
    Returns:
    - DataFrame mit imputierten Werten
    """
    df_imputed = df.copy()
    df_with_features = extract_time_features(df)
    
    # Identifiziere Anomalien
    anomaly_indices = df[df['anomaly'] == 1].index
    
    for anomaly_idx in anomaly_indices:
        # Extrahiere Zeitmerkmale des anomalen Datenpunkts
        anomaly_hour = anomaly_idx.hour
        anomaly_weekday = anomaly_idx.weekday()
        anomaly_month = anomaly_idx.month
        
        # Finde normale Datenpunkte mit ähnlichen Zeitmerkmalen
        normal_points = df_with_features[
            (df_with_features['anomaly'] == 0) & 
            (df_with_features['hour'] == anomaly_hour) & 
            (df_with_features['weekday'] == anomaly_weekday)
        ]
        
        # Wenn keine perfekten Übereinstimmungen gefunden werden, lockere die Bedingungen
        if len(normal_points) < k:
            normal_points = df_with_features[
                (df_with_features['anomaly'] == 0) & 
                (df_with_features['hour'] == anomaly_hour)
            ]
        
        # Wenn immer noch nicht genug gefunden wurden, verwende nur die Stundeneinschränkung
        if len(normal_points) < k:
            normal_points = df_with_features[
                (df_with_features['anomaly'] == 0)
            ]
        
        # Bestimme die k ähnlichsten normalen Datenpunkte basierend auf dem Zeitabstand
        time_diffs = np.abs((normal_points.index - anomaly_idx).total_seconds())
        normal_points['time_diff'] = time_diffs
        similar_points = normal_points.sort_values('time_diff').iloc[:k]
        
        # Berechne den Mittelwert der k ähnlichsten normalen Datenpunkte
        imputed_value = similar_points['meter_reading'].mean()
        
        # Ersetze den anomalen Wert
        df_imputed.loc[anomaly_idx, 'meter_reading'] = imputed_value
        
    return df_imputed

# Funktion zur Visualisierung der Ergebnisse
def visualize_imputation(original_df, imputed_df, window_size=168):  # 168 Stunden = 1 Woche
    """
    Visualisiert die originalen und imputierten Daten in einem bestimmten Zeitfenster
    """
    # Finde einen Zeitraum mit Anomalien
    anomaly_indices = original_df[original_df['anomaly'] == 1].index
    if len(anomaly_indices) > 0:
        center_idx = anomaly_indices[0]
        start_idx = center_idx - timedelta(hours=window_size//2)
        end_idx = center_idx + timedelta(hours=window_size//2)
        
        # Filtere die Daten für das Zeitfenster
        original_window = original_df.loc[start_idx:end_idx]
        imputed_window = imputed_df.loc[start_idx:end_idx]
        
        plt.figure(figsize=(15, 6))
        plt.plot(original_window.index, original_window['meter_reading'], 'b-', label='Original')
        plt.plot(imputed_window.index, imputed_window['meter_reading'], 'g-', label='Imputed')
        
        # Markiere Anomalien
        anomalies = original_window[original_window['anomaly'] == 1]
        plt.scatter(anomalies.index, anomalies['meter_reading'], c='r', s=50, label='Anomalies')
        
        plt.title('Original vs. Imputed Meter Readings')
        plt.xlabel('Timestamp')
        plt.ylabel('Meter Reading')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Hauptfunktion
def main(file_path):
    print("Lade Daten...")
    df = load_data(file_path)
    print(f"Insgesamt {len(df)} Datenpunkte geladen.")
    print(f"Gefundene Anomalien: {df['anomaly'].sum()}")
    
    print("Führe DD-CPI Imputation durch...")
    imputed_df = dd_cpi_imputation(df)
    
    print("Speichere imputierte Daten...")
    imputed_df.reset_index().to_csv('imputed2_meter_readings_118.csv', index=False)
    
    print("Visualisiere die Ergebnisse...")
    visualize_imputation(df, imputed_df)
    
    print("Fertig! Imputierte Daten wurden in 'imputed_meter_readings.csv' gespeichert.")

# Beispielaufruf
if __name__ == "__main__":
    file_path = "id118/Daten/filtered_data_118.csv"  # Passen Sie den Pfad entsprechend an
    main(file_path)