import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
import os

def identify_anomaly_clusters(df_anomalies):
    """Identifiziert zusammenhängende Cluster von Anomalien"""
    anomaly_mask = df_anomalies['anomaly'] == 1
    
    # Erstelle eine neue Spalte, die Clusternummern für aufeinanderfolgende Anomalien enthält
    df_anomalies['cluster'] = (anomaly_mask != anomaly_mask.shift(1)).cumsum() * anomaly_mask
    
    # Hole nur die Anomalie-Cluster (ohne die Nicht-Anomalien mit Cluster-ID 0)
    anomaly_clusters = df_anomalies[df_anomalies['cluster'] > 0].groupby('cluster')
    
    # Erstelle Liste mit den Anomalie-Cluster-Informationen
    clusters = []
    for cluster_id, cluster_data in anomaly_clusters:
        clusters.append({
            'cluster_id': cluster_id,
            'start_idx': cluster_data.index[0],
            'end_idx': cluster_data.index[-1],
            'size': len(cluster_data),
            'timestamps': cluster_data['timestamp'].tolist()
        })
    
    return clusters

def replace_anomalies_incrementally(df_anomalies, df_imputed, percentages=np.arange(0, 101, 10)):
    """
    Ersetzt Anomalien inkrementell in den angegebenen Prozentschritten
    Gibt eine Liste von DataFrames zurück, jeweils mit einem unterschiedlichen Anteil ersetzter Anomalien
    """
    results = []
    clusters = identify_anomaly_clusters(df_anomalies)
    total_clusters = len(clusters)
    
    for percentage in percentages:
        # Erstelle eine Kopie der Anomalie-Daten als Ausgangspunkt
        df_partially_imputed = df_anomalies.copy()
        
        if percentage == 0:
            # Keine Ersetzung bei 0%
            results.append((percentage, df_partially_imputed))
            continue
        
        if percentage == 100:
            # Vollständige Ersetzung bei 100%
            results.append((percentage, df_imputed))
            continue
        
        # Berechne, wie viele Cluster ersetzt werden sollen
        clusters_to_replace = int(np.ceil(total_clusters * percentage / 100))
        
        # Wähle zufällig Cluster zum Ersetzen aus
        np.random.seed(42)  # Für Reproduzierbarkeit
        selected_clusters = np.random.choice(total_clusters, size=clusters_to_replace, replace=False)
        
        # Ersetze die ausgewählten Cluster
        for cluster_idx in selected_clusters:
            cluster = clusters[cluster_idx]
            for timestamp in cluster['timestamps']:
                # Finde den entsprechenden Index in beiden DataFrames
                anomaly_idx = df_anomalies[df_anomalies['timestamp'] == timestamp].index
                imputed_idx = df_imputed[df_imputed['timestamp'] == timestamp].index
                
                if len(anomaly_idx) > 0 and len(imputed_idx) > 0:
                    # Kopiere die imputierten Werte in das teilweise ersetzte DataFrame
                    df_partially_imputed.loc[anomaly_idx, 'meter_reading'] = df_imputed.loc[imputed_idx, 'meter_reading'].values
                    # Setze das Anomalie-Flag auf 0 (nicht anomal)
                    df_partially_imputed.loc[anomaly_idx, 'anomaly'] = 0
        
        results.append((percentage, df_partially_imputed))
    
    return results

def run_nbeats_forecasting(df, building_id, percentage, timestamp_suffix, output_dir="id118neu"):
    """
    Führt die NBEATS-Vorhersage für ein DataFrame durch und speichert die Ergebnisse
    """
    # Erstelle Darts TimeSeries
    series = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
    series_original = series.copy()  

    # Skaliere die Daten
    scaler = Scaler()
    series = scaler.fit_transform(series)

    # Teile in Trainings- und Validierungsdaten
    train, val = series.split_after(0.8)
    val_unscaled = scaler.inverse_transform(val)

    # Früher Stopp für das Training
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # Definiere das NBEATS-Modell
    model = NBEATSModel(
        input_chunk_length=336,
        output_chunk_length=48,
        random_state=42,
        pl_trainer_kwargs={"callbacks": [early_stopping]},
    )

    # Trainiere das Modell
    model.fit(
        series=train,
        val_series=val,
        epochs=50,
        verbose=True
    )

    # Erzeuge Vorhersagen
    forecast_scaled = model.historical_forecasts(
        series,
        start=0.8,                    
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True
    )

    # Rücktransformation der Vorhersagen
    forecast = scaler.inverse_transform(forecast_scaled)

    # Berechne Fehlermetriken
    mae_val = mae(val_unscaled, forecast)
    mean_val = val_unscaled.values().mean()
    mae_percent = (mae_val / mean_val) * 100
    rmse_val = rmse(val_unscaled, forecast)

    print(f"Prozent ersetzter Anomalien: {percentage}%")
    print(f"MAE: {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"Prozentualer MAE: {mae_percent:.2f}%")

    # Erstelle Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    series_original.plot(label="Echte Werte", ax=ax, linewidth=0.5)
    forecast.plot(label="Vorhersage", ax=ax, linewidth=0.5)
    ax.legend()
    ax.set_xlabel("Zeit", fontsize=12)
    ax.set_ylabel("Messwerte", fontsize=12)
    plt.title(f"ID{building_id} - {percentage}% Anomalien ersetzt - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, MAE%: {mae_percent:.2f}%")
    
    # Erstelle Output-Verzeichnis, falls es nicht existiert
    os.makedirs(output_dir, exist_ok=True)
    
    # Speichere Plot
    plt.savefig(f"{output_dir}/forecast_plot_id{building_id}_replaced{percentage}pct_{timestamp_suffix}.png")
    plt.close()

    # Speichere Vorhersagedaten
    forecast_df = forecast.pd_dataframe()
    forecast_df.to_csv(f"{output_dir}/forecast_data_id{building_id}_replaced{percentage}pct_{timestamp_suffix}.csv", index=True)
    
    return {
        'percentage': percentage,
        'mae': mae_val,
        'rmse': rmse_val,
        'mae_percent': mae_percent
    }

def main():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_with_anomalies = "id439neu/filtered_data_building_439_filled_mehr_1.csv"  # Pfad zur Datei mit Anomalien
    file_with_imputations = "id439neu/imputed_meter_readings_439_CPI_mehr_1.csv"  # Pfad zur Datei mit imputierten Anomalien
    df_anomalies = pd.read_csv(file_with_anomalies, parse_dates=["timestamp"])
    df_imputed = pd.read_csv(file_with_imputations, parse_dates=["timestamp"])

    building_id_anomalies = int(df_anomalies["building_id"].iloc[0])
    building_id_imputed = int(df_imputed["building_id"].iloc[0])
    if building_id_anomalies != building_id_imputed:
        print(f"WARNUNG: Die Dateien haben unterschiedliche Gebäude-IDs: {building_id_anomalies} vs {building_id_imputed}")
        return
    building_id = building_id_anomalies

    output_dir = f"Forecast/ID{building_id}_Analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    partially_imputed_dataframes = replace_anomalies_incrementally(df_anomalies, df_imputed)
    anomaly_clusters = identify_anomaly_clusters(df_anomalies)
    cluster_info = pd.DataFrame([
        {
            'cluster_id': cluster['cluster_id'],
            'start_time': min(cluster['timestamps']),
            'end_time': max(cluster['timestamps']),
            'size': cluster['size']
        }
        for cluster in anomaly_clusters
    ])
    
    cluster_info.to_csv(f"{output_dir}/anomaly_clusters_info.csv", index=False)
    print(f"Identifizierte Anomalie-Cluster: {len(anomaly_clusters)}")
    print(f"Gesamtzahl der Anomalien: {df_anomalies['anomaly'].sum()}")
    results = []
    for percentage, df in partially_imputed_dataframes:
        result = run_nbeats_forecasting(df, building_id, percentage, timestamp, output_dir)
        results.append(result)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/forecast_metrics_summary.csv", index=False)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results_df['percentage'], results_df['mae'], marker='o', label='MAE')
    plt.plot(results_df['percentage'], results_df['rmse'], marker='s', label='RMSE')
    plt.xlabel('Prozent ersetzter Anomalien')
    plt.ylabel('Fehlermetriken')
    plt.title(f'Einfluss des Ersetzens von Anomalien auf die Vorhersagegenauigkeit - ID{building_id}')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(results_df['percentage'], results_df['mae_percent'], marker='o', color='green')
    plt.xlabel('Prozent ersetzter Anomalien')
    plt.ylabel('MAE in Prozent')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_vs_replacement_percentage.png")
    
    print(f"Analyse abgeschlossen. Ergebnisse wurden in {output_dir} gespeichert.")

if __name__ == "__main__":
    main()
