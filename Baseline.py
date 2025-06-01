import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse, mape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FiveDayForecastComparison:
    def __init__(self, filepath):
        self.filepath = filepath
        self.forecast_horizon = 24 * 5  # 5 Tage = 120 Stunden
        self.results = {}
        
    def load_and_prepare_data(self):
        print("Daten laden und vorbereiten...")
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        self.series_original = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
        
        # Aufteilen: Training bis 5 Tage vor Ende
        total_length = len(self.series_original)
        train_end = total_length - self.forecast_horizon
        
        self.train_series = self.series_original[:train_end]
        self.test_series = self.series_original[train_end:]
        
        print(f"Gesamte Zeitreihe: {total_length} Punkte")
        print(f"Training: {len(self.train_series)} Punkte")
        print(f"Test (5 Tage): {len(self.test_series)} Punkte")
        print(f"Vorhersagehorizont: {self.forecast_horizon} Stunden")
        
    def evaluate_model(self, predictions, model_name):
        """Berechnet Metriken für ein Modell"""
        mae_val = mae(self.test_series, predictions)
        rmse_val = rmse(self.test_series, predictions)
        mape_val = mape(self.test_series, predictions)
        
        mean_actual = self.test_series.values().mean()
        mae_percent = (mae_val / mean_actual) * 100
        
        result = {
            'model': model_name,
            'mae': mae_val,
            'mae_percent': mae_percent,
            'rmse': rmse_val,
            'mape': mape_val,
            'predictions': predictions
        }
        
        self.results[model_name] = result
        
        print(f"\n{model_name} Ergebnisse:")
        print(f"MAE: {mae_val:.4f}")
        print(f"MAE%: {mae_percent:.2f}%")
        print(f"RMSE: {rmse_val:.4f}")
        print(f"MAPE: {mape_val:.2f}%")
        
        return result
    
    def moving_average_5day_forecast(self, window_days=5):
        """Moving Average für 5-Tage Vorhersage"""
        print(f"\n=== Moving Average ({window_days} Tage Fenster) ===")
        
        window = 24 * window_days
        last_values = self.train_series.values()[-window:].flatten()
        avg_value = np.mean(last_values)
        
        # Konstante Vorhersage für 5 Tage
        ma_values = np.full(self.forecast_horizon, avg_value)
        
        forecast_index = self.test_series.time_index
        ma_forecast = TimeSeries.from_times_and_values(
            times=forecast_index,
            values=ma_values
        )
        
        return self.evaluate_model(ma_forecast, f"Moving_Average_{window_days}d")
    
    def nbeats_5day_forecast(self):
        """NBEATS für 5-Tage Vorhersage"""
        print(f"\n=== NBEATS (ohne Kovariaten) ===")
        
        # Daten skalieren
        scaler = Scaler()
        train_scaled = scaler.fit_transform(self.train_series)
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor="train_loss", patience=10, mode="min", verbose=True
        )
        
        # NBEATS Modell
        nbeats_model = NBEATSModel(
            input_chunk_length=336,  # 14 Tage Input
            output_chunk_length=120, # 5 Tage Output
            num_stacks=3,
            num_blocks=3,
            num_layers=4,
            layer_widths=256,
            trend_polynomial_degree=3,
            dropout=0.1,
            activation="ReLU",
            n_epochs=50,
            batch_size=32,
            random_state=42,
            force_reset=True,
            pl_trainer_kwargs={
                "enable_progress_bar": True,
                "callbacks": [early_stopping],
                "logger": False,
                "enable_checkpointing": False
            }
        )
        
        # Training
        print("NBEATS Training...")
        nbeats_model.fit(series=train_scaled, verbose=True)
        
        # Vorhersage
        print("NBEATS Vorhersage...")
        nbeats_pred_scaled = nbeats_model.predict(
            n=self.forecast_horizon,
            series=train_scaled
        )
        
        # Rück-skalierung
        nbeats_pred = scaler.inverse_transform(nbeats_pred_scaled)
        
        return self.evaluate_model(nbeats_pred, "NBEATS")
    
    def nbeatsx_5day_forecast(self):
        """NBEATSx für 5-Tage Vorhersage"""
        print(f"\n=== NBEATSx (mit Kovariaten) ===")
        
        # Kovariaten vorbereiten
        print("Erstelle Kovariaten...")
        hour_cov = datetime_attribute_timeseries(
            self.series_original, attribute="hour", one_hot=False
        )
        day_cov = datetime_attribute_timeseries(
            self.series_original, attribute="dayofweek", one_hot=False
        )
        month_cov = datetime_attribute_timeseries(
            self.series_original, attribute="month", one_hot=False
        )
        
        covariates = hour_cov.stack(day_cov).stack(month_cov)
        train_covariates = covariates[:len(self.train_series) + self.forecast_horizon]
        
        # Daten skalieren
        scaler = Scaler()
        train_scaled = scaler.fit_transform(self.train_series)
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor="train_loss", patience=10, mode="min", verbose=True
        )
        
        # NBEATSx Modell
        nbeatsx_model = NBEATSModel(
            input_chunk_length=336,  # 14 Tage Input
            output_chunk_length=120, # 5 Tage Output
            expansion_coefficient_dim=5,  # Für Kovariaten
            num_stacks=3,
            num_blocks=3,
            num_layers=4,
            layer_widths=256,
            trend_polynomial_degree=3,
            dropout=0.1,
            activation="ReLU",
            n_epochs=50,
            batch_size=32,
            random_state=42,
            force_reset=True,
            pl_trainer_kwargs={
                "enable_progress_bar": True,
                "callbacks": [early_stopping],
                "logger": False,
                "enable_checkpointing": False
            }
        )
        
        # Training mit Kovariaten
        print("NBEATSx Training...")
        nbeatsx_model.fit(
            series=train_scaled, 
            past_covariates=train_covariates,
            verbose=True
        )
        
        # Vorhersage mit Kovariaten
        print("NBEATSx Vorhersage...")
        nbeatsx_pred_scaled = nbeatsx_model.predict(
            n=self.forecast_horizon,
            series=train_scaled,
            past_covariates=covariates
        )
        
        # Rück-skalierung
        nbeatsx_pred = scaler.inverse_transform(nbeatsx_pred_scaled)
        
        return self.evaluate_model(nbeatsx_pred, "NBEATSx")
    
    def run_all_models(self):
        """Führt alle drei Modelle aus"""
        print("=" * 70)
        print("         5-TAGE VORHERSAGE VERGLEICH")
        print("    Moving Average vs NBEATS vs NBEATSx")
        print("=" * 70)
        
        self.load_and_prepare_data()
        
        # Alle Modelle ausführen
        self.moving_average_5day_forecast(window_days=5)
        self.nbeats_5day_forecast()
        self.nbeatsx_5day_forecast()
        
        return self.results
    
    def print_comparison(self):
        """Druckt Vergleichstabelle"""
        print("\n" + "=" * 80)
        print("                    5-TAGE VORHERSAGE ERGEBNISSE")
        print("=" * 80)
        print(f"{'Model':<20} {'MAE':<12} {'MAE%':<12} {'RMSE':<12} {'MAPE':<12}")
        print("-" * 80)
        
        # Sortiere nach MAE%
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['mae_percent'])
        
        for model_name, result in sorted_results:
            print(f"{model_name:<20} {result['mae']:<12.4f} {result['mae_percent']:<12.2f} "
                  f"{result['rmse']:<12.4f} {result['mape']:<12.2f}")
        
        print("-" * 80)
        
        # Vergleichsanalyse
        best_model = sorted_results[0][0]
        best_mae = sorted_results[0][1]['mae_percent']
        worst_model = sorted_results[-1][0]
        worst_mae = sorted_results[-1][1]['mae_percent']
        
        print(f"Bestes Modell: {best_model} (MAE%: {best_mae:.2f}%)")
        print(f"Schlechtestes Modell: {worst_model} (MAE%: {worst_mae:.2f}%)")
        print(f"Verbesserung: {worst_mae - best_mae:.2f} Prozentpunkte")
        
        # Spezifische Vergleiche
        if "NBEATSx" in self.results and "NBEATS" in self.results:
            nbeatsx_mae = self.results["NBEATSx"]["mae_percent"]
            nbeats_mae = self.results["NBEATS"]["mae_percent"]
            improvement = nbeats_mae - nbeatsx_mae
            print(f"\nNBEATSx vs NBEATS: {improvement:.2f} Prozentpunkte Verbesserung")
        
        if "Moving_Average_5d" in self.results and best_model != "Moving_Average_5d":
            ma_mae = self.results["Moving_Average_5d"]["mae_percent"]
            deep_learning_improvement = ma_mae - best_mae
            print(f"Deep Learning vs Moving Average: {deep_learning_improvement:.2f} Prozentpunkte Verbesserung")
        
        print("=" * 80)
    
    def plot_5day_comparison(self):
        """Erstellt umfassende Visualisierung"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Hauptplot: Alle Vorhersagen
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Gesamtübersicht
        context_length = min(336, len(self.train_series))  # 14 Tage Kontext
        train_context = self.train_series[-context_length:]
        
        train_context.plot(label="Training (letzte 14 Tage)", ax=ax1, linewidth=1, alpha=0.7, color='gray')
        self.test_series.plot(label="Echte Werte (5 Tage)", ax=ax1, linewidth=3, color='black')
        
        colors = ['red', 'blue', 'green']
        for i, (model_name, result) in enumerate(self.results.items()):
            result['predictions'].plot(
                label=f"{model_name} (MAE%: {result['mae_percent']:.2f}%)", 
                ax=ax1, linewidth=2, linestyle='--', color=colors[i % len(colors)]
            )
        
        ax1.set_title("5-Tage Vorhersage Vergleich - Gesamtübersicht")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Energieverbrauch")
        
        # Plot 2: Nur Vorhersagebereich (Zoom)
        self.test_series.plot(label="Echte Werte", ax=ax2, linewidth=3, color='black')
        
        for i, (model_name, result) in enumerate(self.results.items()):
            result['predictions'].plot(
                label=f"{model_name}", 
                ax=ax2, linewidth=2, linestyle='--', color=colors[i % len(colors)]
            )
        
        ax2.set_title("5-Tage Vorhersage - Detailansicht")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel("Energieverbrauch")
        ax2.set_xlabel("Zeit")
        
        # Plot 3: Metriken Vergleich
        models = list(self.results.keys())
        mae_percents = [self.results[model]['mae_percent'] for model in models]
        
        bars = ax3.bar(models, mae_percents, color=colors[:len(models)], alpha=0.7, edgecolor='black')
        
        for bar, value in zip(bars, mae_percents):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Bestes Modell hervorheben
        best_idx = mae_percents.index(min(mae_percents))
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        ax3.set_ylabel('MAE% (Mittlerer Absoluter Fehler in %)')
        ax3.set_title('Performance Vergleich: MAE%')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 4: Tägliche Performance
        actual_values = self.test_series.values().flatten()
        
        # Teile 5-Tage-Periode in einzelne Tage
        daily_mae = {}
        for model_name, result in self.results.items():
            pred_values = result['predictions'].values().flatten()
            daily_errors = []
            
            for day in range(5):
                day_start = day * 24
                day_end = (day + 1) * 24
                day_actual = actual_values[day_start:day_end]
                day_pred = pred_values[day_start:day_end]
                day_mae = np.mean(np.abs(day_actual - day_pred))
                daily_errors.append(day_mae)
            
            daily_mae[model_name] = daily_errors
        
        days = ['Tag 1', 'Tag 2', 'Tag 3', 'Tag 4', 'Tag 5']
        x = np.arange(len(days))
        width = 0.25
        
        for i, (model_name, errors) in enumerate(daily_mae.items()):
            ax4.bar(x + i * width, errors, width, label=model_name, color=colors[i % len(colors)], alpha=0.7)
        
        ax4.set_xlabel('Vorhersagetag')
        ax4.set_ylabel('MAE')
        ax4.set_title('Tägliche Performance (MAE pro Tag)')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(days)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"5Day_Forecast_Comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detailed_error_analysis(self):
        """Erstellt detaillierte Fehleranalyse"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        actual_values = self.test_series.values().flatten()
        colors = ['red', 'blue', 'green']
        
        for i, (model_name, result) in enumerate(self.results.items()):
            ax = axes[i]
            
            pred_values = result['predictions'].values().flatten()
            errors = actual_values - pred_values
            
            # Fehler über Zeit
            ax.plot(errors, color=colors[i], linewidth=1.5, label='Fehler')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax.fill_between(range(len(errors)), errors, 0, alpha=0.3, color=colors[i])
            
            # Tage markieren
            for day in range(1, 5):
                ax.axvline(x=day*24, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f"{model_name}\nMAE%: {result['mae_percent']:.2f}%")
            ax.set_xlabel("Stunden")
            ax.set_ylabel("Vorhersagefehler")
            ax.grid(True, alpha=0.3)
            
            # Statistiken
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax.text(0.02, 0.98, f"μ: {mean_error:.3f}\nσ: {std_error:.3f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.suptitle("5-Tage Vorhersage: Fehleranalyse über Zeit", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"5Day_Error_Analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_comparison(self):
        """Führt die komplette Analyse durch"""
        # Alle Modelle ausführen
        self.run_all_models()
        
        # Ergebnisse anzeigen
        self.print_comparison()
        
        # Visualisierungen erstellen
        print("\nErstelle Visualisierungen...")
        self.plot_5day_comparison()
        self.plot_detailed_error_analysis()
        
        print("\n5-Tage Vorhersage Vergleich abgeschlossen!")
        return self.results

if __name__ == "__main__":
    comparison = FiveDayForecastComparison(
        filepath="id118/imputed_meter_readings_118_CPI.csv"
    )
    
    results = comparison.run_complete_comparison()
    
    print(f"\nFazit:")
    best_model = min(results.items(), key=lambda x: x[1]['mae_percent'])
    print(f"Bestes Modell für 5-Tage Vorhersage: {best_model[0]}")
    print(f"Erreichte Performance: MAE% = {best_model[1]['mae_percent']:.2f}%")