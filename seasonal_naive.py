import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, rmse, mape
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Simple5DayMovingAverage:
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
        
    def create_5day_moving_average_forecast(self):
        """
        5-Day Moving Average: Durchschnitt der letzten 5 Tage (120 Stunden)
        """
        print("\nErstelle 5-Day Moving Average Vorhersage...")
        
        window_size = 24 * 5  # 120 Stunden = 5 Tage
        train_values = self.train_series.values().flatten()
        
        # Berechne Durchschnitt der letzten 5 Tage
        last_5_days = train_values[-window_size:]
        avg_value = np.mean(last_5_days)
        
        print(f"Durchschnitt der letzten 5 Tage: {avg_value:.4f}")
        
        # Konstante Vorhersage für alle Zeitpunkte
        forecast_values = np.full(self.forecast_horizon, avg_value)
        
        # TimeSeries aus Vorhersagen erstellen
        forecast_index = self.test_series.time_index
        ma_forecast = TimeSeries.from_times_and_values(
            times=forecast_index,
            values=forecast_values
        )
        
        return ma_forecast
    
    def evaluate_forecast(self, forecast):
        """Evaluiert die Vorhersage"""
        mae_val = mae(self.test_series, forecast)
        rmse_val = rmse(self.test_series, forecast)
        mape_val = mape(self.test_series, forecast)
        
        mean_actual = self.test_series.values().mean()
        mae_percent = (mae_val / mean_actual) * 100
        
        result = {
            'model': '5Day_Moving_Average',
            'mae': mae_val,
            'mae_percent': mae_percent,
            'rmse': rmse_val,
            'mape': mape_val,
            'forecast': forecast
        }
        
        print(f"\n5-Day Moving Average Ergebnisse:")
        print(f"MAE: {mae_val:.4f}")
        print(f"MAE%: {mae_percent:.2f}%")
        print(f"RMSE: {rmse_val:.4f}")
        print(f"MAPE: {mape_val:.2f}%")
        
        return result
    
    def plot_forecast(self, result):
        """Erstellt Plot der Vorhersage"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Gesamtübersicht
        self.series_original.plot(label="Echte Werte (komplett)", ax=ax1, linewidth=0.8, alpha=0.7, color='blue')
        result['forecast'].plot(
            label=f"5-Day Moving Average (MAE%: {result['mae_percent']:.2f}%)", 
            ax=ax1, linewidth=2, color='red', linestyle='--'
        )
        
        # Markiere Trainings-/Test-Grenze
        split_point = len(self.train_series)
        split_time = self.series_original.time_index[split_point]
        ax1.axvline(x=split_time, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Train/Test Split')
        
        # Markiere die letzten 5 Tage die für den Durchschnitt verwendet wurden
        window_start_time = self.series_original.time_index[split_point - 120]  # 120h = 5 Tage
        ax1.axvspan(window_start_time, split_time, alpha=0.2, color='orange', label='5-Tage Fenster')
        
        ax1.legend()
        ax1.set_title("5-Day Moving Average - Vorhersage der letzten 20%")
        ax1.set_ylabel("Energieverbrauch")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detailansicht der letzten Periode
        context_length = min(240, len(self.train_series))  # Letzte 10 Tage als Kontext
        train_context = self.train_series[-context_length:]
        
        train_context.plot(label="Training (letzte 10 Tage)", ax=ax2, linewidth=1, alpha=0.7, color='blue')
        self.test_series.plot(label="Echte Werte (Test)", ax=ax2, linewidth=2, color='black')
        result['forecast'].plot(
            label=f"5-Day Moving Average (MAE%: {result['mae_percent']:.2f}%)", 
            ax=ax2, linewidth=2, color='red', linestyle='--'
        )
        
        # Markiere das 5-Tage Fenster im Detail
        window_start_idx = len(train_context) - 120
        if window_start_idx >= 0:
            ax2.axvspan(train_context.time_index[window_start_idx], train_context.time_index[-1], 
                       alpha=0.2, color='orange', label='5-Tage Durchschnitt')
        
        ax2.legend()
        ax2.set_title("5-Day Moving Average - Detailansicht")
        ax2.set_ylabel("Energieverbrauch")
        ax2.set_xlabel("Zeit")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"5Day_Moving_Average_Forecast_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_5day_window(self):
        """Analysiert das 5-Tage Fenster das für den Durchschnitt verwendet wird"""
        window_size = 24 * 5
        train_values = self.train_series.values().flatten()
        last_5_days = train_values[-window_size:]
        
        print(f"\nAnalyse des 5-Tage Fensters:")
        print(f"Durchschnitt: {np.mean(last_5_days):.4f}")
        print(f"Minimum: {np.min(last_5_days):.4f}")
        print(f"Maximum: {np.max(last_5_days):.4f}")
        print(f"Standardabweichung: {np.std(last_5_days):.4f}")
        
        # Tägliche Durchschnitte
        daily_averages = []
        for day in range(5):
            day_start = day * 24
            day_end = (day + 1) * 24
            day_avg = np.mean(last_5_days[day_start:day_end])
            daily_averages.append(day_avg)
            print(f"Tag {day+1} Durchschnitt: {day_avg:.4f}")
        
        return daily_averages
    
    def run_5day_moving_average_forecast(self):
        """Führt die komplette 5-Day Moving Average Vorhersage durch"""
        print("=" * 60)
        print("        5-DAY MOVING AVERAGE BASELINE")
        print("=" * 60)
        
        # Daten laden
        self.load_and_prepare_data()
        
        # Analysiere das 5-Tage Fenster
        daily_averages = self.analyze_5day_window()
        
        # Vorhersage erstellen
        forecast = self.create_5day_moving_average_forecast()
        
        # Evaluierung
        result = self.evaluate_forecast(forecast)
        
        # Plot erstellen
        print("\nErstelle Visualisierung...")
        self.plot_forecast(result)
        
        print("\n5-Day Moving Average Baseline abgeschlossen!")
        return result

if __name__ == "__main__":
    ma_baseline = Simple5DayMovingAverage(
        filepath="id118/imputed_meter_readings_118_CPI.csv"
    )
    
    result = ma_baseline.run_5day_moving_average_forecast()
    
    print(f"\n5-Day Moving Average Baseline Ergebnis:")
    print(f"MAE% = {result['mae_percent']:.2f}%")
    print("Dieses Ergebnis kann direkt mit NBEATS und NBEATSx verglichen werden.")