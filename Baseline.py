import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, rmse, mape
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Simple7DayMovingAverage:
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load_and_prepare_data(self):
        print("Daten laden und vorbereiten...")
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        self.series_original = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
        
        # 80/20 Split wie bei NBEATS/NBEATSx
        split_point = int(0.8 * len(self.series_original))
        self.train_series = self.series_original[:split_point]
        self.test_series = self.series_original[split_point:]
        
        # Anzahl der vorherzusagenden Punkte (komplette 20%)
        self.forecast_horizon = len(self.test_series)
        
        print(f"Gesamte Zeitreihe: {len(self.series_original)} Punkte")
        print(f"Training (80%): {len(self.train_series)} Punkte")
        print(f"Test (20%): {len(self.test_series)} Punkte")
        print(f"Vorhersagehorizont: {self.forecast_horizon} Stunden")
        
    def create_rolling_7day_moving_average_forecast(self):
        """
        Rolling 7-Day Moving Average: 
        - Verwende echte Werte wenn verfügbar
        - Aktualisiere Durchschnitt alle 7 Tage basierend auf echten Werten
        """
        print("\nErstelle Rolling 7-Day Moving Average Vorhersage...")
        
        window_size = 24 * 7  # 168 Stunden = 7 Tage
        chunk_size = 24 * 7   # Alle 7 Tage neuen Durchschnitt berechnen
        
        # Kombiniere Training und Test für rolling forecast
        all_values = np.concatenate([
            self.train_series.values().flatten(),
            self.test_series.values().flatten()
        ])
        
        train_end_idx = len(self.train_series)
        forecast_values = []
        
        print(f"Verwende Rolling Window von {window_size} Stunden")
        print(f"Aktualisiere Durchschnitt alle {chunk_size} Stunden")
        
        for i in range(self.forecast_horizon):
            current_idx = train_end_idx + i
            
            # Bestimme ob wir einen neuen Durchschnitt berechnen müssen
            if i % chunk_size == 0:
                # Berechne neuen Durchschnitt basierend auf den letzten 7 Tagen
                window_start = current_idx - window_size
                window_end = current_idx
                
                # Verwende echte Werte (aus Training + bereits beobachteten Test-Werten)
                window_values = all_values[window_start:window_end]
                current_average = np.mean(window_values)
                
                print(f"Stunde {i}: Neuer 7-Tage Durchschnitt berechnet: {current_average:.4f}")
                print(f"  Basierend auf Werten von Index {window_start} bis {window_end-1}")
            
            # Verwende den aktuellen Durchschnitt für die Vorhersage
            forecast_values.append(current_average)
        
        # TimeSeries aus Vorhersagen erstellen
        forecast_index = self.test_series.time_index
        rolling_ma_forecast = TimeSeries.from_times_and_values(
            times=forecast_index,
            values=forecast_values
        )
        
        return rolling_ma_forecast
    
    def evaluate_forecast(self, forecast):
        """Evaluiert die Vorhersage"""
        mae_val = mae(self.test_series, forecast)
        rmse_val = rmse(self.test_series, forecast)
        mape_val = mape(self.test_series, forecast)
        
        mean_actual = self.test_series.values().mean()
        mae_percent = (mae_val / mean_actual) * 100
        
        result = {
            'model': 'Rolling_7Day_Moving_Average',
            'mae': mae_val,
            'mae_percent': mae_percent,
            'rmse': rmse_val,
            'mape': mape_val,
            'forecast': forecast
        }
        
        print(f"\nRolling 7-Day Moving Average Ergebnisse:")
        print(f"MAE: {mae_val:.4f}")
        print(f"MAE%: {mae_percent:.2f}%")
        print(f"RMSE: {rmse_val:.4f}")
        print(f"MAPE: {mape_val:.2f}%")
        
        return result
    
    def plot_forecast_with_updates(self, result):
        """Erstellt Plot der Vorhersage mit Markierungen für Updates"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: Gesamtübersicht
        self.series_original.plot(label="Echte Werte (komplett)", ax=ax1, linewidth=0.8, alpha=0.7, color='blue')
        result['forecast'].plot(
            label=f"Rolling 7-Day MA (MAE%: {result['mae_percent']:.2f}%)", 
            ax=ax1, linewidth=2, color='red', linestyle='--'
        )
        
        # Markiere Trainings-/Test-Grenze
        split_point = len(self.train_series)
        split_time = self.series_original.time_index[split_point]
        ax1.axvline(x=split_time, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Train/Test Split')
        
        # Markiere Update-Punkte (alle 7 Tage)
        chunk_size = 24 * 7
        for i in range(0, self.forecast_horizon, chunk_size):
            update_time = self.test_series.time_index[i]
            ax1.axvline(x=update_time, color='orange', linestyle=':', alpha=0.7)
        
        ax1.axvline(x=self.test_series.time_index[0], color='orange', linestyle=':', alpha=0.7, label='MA Updates')
        
        ax1.legend()
        ax1.set_title("Rolling 7-Day Moving Average - Vorhersage mit Updates alle 7 Tage")
        ax1.set_ylabel("Energieverbrauch")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detailansicht mit Segmenten
        context_length = min(240, len(self.train_series))  # Letzte 10 Tage als Kontext
        train_context = self.train_series[-context_length:]
        
        train_context.plot(label="Training (letzte 10 Tage)", ax=ax2, linewidth=1, alpha=0.7, color='blue')
        self.test_series.plot(label="Echte Werte (Test)", ax=ax2, linewidth=2, color='black')
        
        # Zeige Vorhersage in Segmenten (verschiedene Farben für verschiedene MA-Perioden)
        colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
        chunk_size = 24 * 7
        
        for segment in range(0, self.forecast_horizon, chunk_size):
            segment_end = min(segment + chunk_size, self.forecast_horizon)
            segment_forecast = result['forecast'][segment:segment_end]
            
            color_idx = (segment // chunk_size) % len(colors)
            segment_forecast.plot(
                label=f"MA Segment {segment//chunk_size + 1}", 
                ax=ax2, linewidth=2, linestyle='--', color=colors[color_idx]
            )
        
        # Markiere Update-Punkte
        for i in range(0, self.forecast_horizon, chunk_size):
            if i < len(self.test_series.time_index):
                update_time = self.test_series.time_index[i]
                ax2.axvline(x=update_time, color='orange', linestyle=':', alpha=0.7)
        
        ax2.legend()
        ax2.set_title("Rolling 7-Day Moving Average - Segmente und Updates")
        ax2.set_ylabel("Energieverbrauch")
        ax2.set_xlabel("Zeit")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"Rolling_7Day_Moving_Average_Forecast_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_rolling_segments(self):
        """Analysiert die verschiedenen Rolling-Segmente"""
        window_size = 24 * 7
        chunk_size = 24 * 7
        
        # Kombiniere Training und Test
        all_values = np.concatenate([
            self.train_series.values().flatten(),
            self.test_series.values().flatten()
        ])
        train_end_idx = len(self.train_series)
        
        print(f"\nAnalyse der Rolling Segmente:")
        
        segment_averages = []
        for i in range(0, self.forecast_horizon, chunk_size):
            current_idx = train_end_idx + i
            
            # Berechne Durchschnitt für dieses Segment
            window_start = current_idx - window_size
            window_end = current_idx
            window_values = all_values[window_start:window_end]
            segment_avg = np.mean(window_values)
            
            segment_num = i // chunk_size + 1
            days_range = f"Tag {i//24 + 1}-{min((i + chunk_size)//24, self.forecast_horizon//24)}"
            
            print(f"Segment {segment_num} ({days_range}): Durchschnitt = {segment_avg:.4f}")
            segment_averages.append(segment_avg)
        
        return segment_averages
    
    def run_rolling_7day_moving_average_forecast(self):
        """Führt die komplette Rolling 7-Day Moving Average Vorhersage durch"""
        print("=" * 70)
        print("        ROLLING 7-DAY MOVING AVERAGE BASELINE")
        print("     (Updates alle 7 Tage basierend auf echten Werten)")
        print("=" * 70)
        
        # Daten laden
        self.load_and_prepare_data()
        
        # Analysiere die Segmente
        segment_averages = self.analyze_rolling_segments()
        
        # Vorhersage erstellen
        forecast = self.create_rolling_7day_moving_average_forecast()
        
        # Evaluierung
        result = self.evaluate_forecast(forecast)
        
        # Plot erstellen
        print("\nErstelle Visualisierung...")
        self.plot_forecast_with_updates(result)
        
        print("\nRolling 7-Day Moving Average Baseline abgeschlossen!")
        return result

if __name__ == "__main__":
    rolling_ma_baseline = Simple7DayMovingAverage(
        filepath="id254/imputed_meter_readings_254_CPI_mehr_1.csv"
    )
    
    result = rolling_ma_baseline.run_rolling_7day_moving_average_forecast()
    
    print(f"\nRolling 7-Day Moving Average Baseline Ergebnis:")
    print(f"MAE% = {result['mae_percent']:.2f}%")
    print("Dieses Ergebnis kann direkt mit NBEATS und NBEATSx verglichen werden.")