def seasonal_naive_forecast(self):
        """Seasonal Naive: Wiederholung von vor 168 Stunden (1 Woche)"""
        print("\n=== Seasonal Naive (168h Saison) ===")
        
        season_length = 24 * 7  # 168 Stunden = 1 Woche
        train_values = self.train_series.values().flatten()
        
        # Vorhersagewerte aus den letzten season_length Werten
        forecast_values = []
        
        for i in range(self.forecast_horizon):
            # Index des entsprechenden Wertes vor season_length Stunden
            source_idx = len(train_values) - season_length + (i % season_length)
            forecast_values.append(train_values[source_idx])
        
        # TimeSeries aus Vorhersagen erstellen
        forecast_index = self.test_series.time_index
        seasonal_forecast = TimeSeries.from_times_and_values(
            times=forecast_index,
            values=forecast_values
        )
        
        return self.evaluate_model(seasonal_forecast, "Seasonal_Naive")
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

class FourModelComparison:
    def __init__(self, filepath):
        self.filepath = filepath
        self.results = {}
        
    def load_and_prepare_data(self):
        print("Daten laden und vorbereiten...")
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        self.series_original = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
        
        # 80/20 Split
        split_point = int(0.8 * len(self.series_original))
        self.train_series = self.series_original[:split_point]
        self.test_series = self.series_original[split_point:]
        self.forecast_horizon = len(self.test_series)
        
        print(f"Gesamte Zeitreihe: {len(self.series_original)} Punkte")
        print(f"Training (80%): {len(self.train_series)} Punkte")
        print(f"Test (20%): {len(self.test_series)} Punkte")
        print(f"Vorhersagehorizont: {self.forecast_horizon} Stunden")
        
    def evaluate_model(self, predictions, model_name):
        """Berechnet Metriken für ein Modell"""
        # Sicherstellen, dass Vorhersagen und Test-Daten gleiche Länge haben
        min_length = min(len(self.test_series), len(predictions))
        test_subset = self.test_series[:min_length]
        pred_subset = predictions[:min_length]
        
        mae_val = mae(test_subset, pred_subset)
        rmse_val = rmse(test_subset, pred_subset)
        mape_val = mape(test_subset, pred_subset)
        
        mean_actual = test_subset.values().mean()
        mae_percent = (mae_val / mean_actual) * 100
        
        result = {
            'model': model_name,
            'mae': mae_val,
            'mae_percent': mae_percent,
            'rmse': rmse_val,
            'mape': mape_val,
            'predictions': pred_subset  # Verwende angepasste Länge
        }
        
        self.results[model_name] = result
        
        print(f"\n{model_name} Ergebnisse:")
        print(f"Evaluiert auf {min_length} Datenpunkten")
        print(f"MAE: {mae_val:.4f}")
        print(f"MAE%: {mae_percent:.2f}%")
        print(f"RMSE: {rmse_val:.4f}")
        print(f"MAPE: {mape_val:.2f}%")
        
        return result
    
    def rolling_7day_moving_average(self):
        """Rolling 7-Day Moving Average"""
        print("\n=== Rolling 7-Day Moving Average ===")
        
        window_size = 24 * 7  # 168 Stunden
        chunk_size = 24 * 7   # Updates alle 7 Tage
        
        # Kombiniere Training und Test
        all_values = np.concatenate([
            self.train_series.values().flatten(),
            self.test_series.values().flatten()
        ])
        
        train_end_idx = len(self.train_series)
        forecast_values = []
        
        for i in range(self.forecast_horizon):
            current_idx = train_end_idx + i
            
            if i % chunk_size == 0:
                window_start = current_idx - window_size
                window_end = current_idx
                window_values = all_values[window_start:window_end]
                current_average = np.mean(window_values)
            
            forecast_values.append(current_average)
        
        forecast_index = self.test_series.time_index
        ma_forecast = TimeSeries.from_times_and_values(
            times=forecast_index,
            values=forecast_values
        )
        
        return self.evaluate_model(ma_forecast, "7Day_Moving_Average")
    
    def seasonal_naive_forecast(self):
        """Seasonal Naive: Wiederholung von vor 168 Stunden (1 Woche)"""
        print("\n=== Seasonal Naive (168h Saison) ===")
        
        season_length = 24 * 7  # 168 Stunden = 1 Woche
        train_values = self.train_series.values().flatten()
        
        # Vorhersagewerte aus den letzten season_length Werten
        forecast_values = []
        
        for i in range(self.forecast_horizon):
            # Index des entsprechenden Wertes vor season_length Stunden
            source_idx = len(train_values) - season_length + (i % season_length)
            forecast_values.append(train_values[source_idx])
        
        # TimeSeries aus Vorhersagen erstellen
        forecast_index = self.test_series.time_index
        seasonal_forecast = TimeSeries.from_times_and_values(
            times=forecast_index,
            values=forecast_values
        )
        
        return self.evaluate_model(seasonal_forecast, "Seasonal_Naive")
    
    def nbeats_forecast(self):
        """NBEATS Vorhersage mit historical_forecasts"""
        print("\n=== NBEATS (ohne Kovariaten) ===")
        
        # Daten skalieren
        scaler = Scaler()
        train_scaled = scaler.fit_transform(self.train_series)
        series_scaled = scaler.fit_transform(self.series_original)
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor="train_loss", patience=10, mode="min", verbose=True
        )
        
        # NBEATS Modell
        nbeats_model = NBEATSModel(
            input_chunk_length=336,  # 14 Tage
            output_chunk_length=168, # 1 Woche (anstatt 48h)
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
        
        print("NBEATS Training...")
        nbeats_model.fit(series=train_scaled, verbose=True)
        
        print("NBEATS Historical Forecasts...")
        nbeats_pred_scaled = nbeats_model.historical_forecasts(
            series_scaled,
            start=0.8,
            forecast_horizon=1,  # 1-Schritt Vorhersagen
            stride=1,
            retrain=False,
            verbose=True
        )
        
        nbeats_pred = scaler.inverse_transform(nbeats_pred_scaled)
        
        return self.evaluate_model(nbeats_pred, "NBEATS")
    
    def nbeatsx_forecast(self):
        """NBEATSx Vorhersage mit historical_forecasts"""
        print("\n=== NBEATSx (mit Kovariaten) ===")
        
        # Kovariaten erstellen
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
        
        # Kovariaten für Training
        split_point = int(0.8 * len(self.series_original))
        train_covariates = covariates[:split_point]
        
        # Daten skalieren
        scaler = Scaler()
        train_scaled = scaler.fit_transform(self.train_series)
        series_scaled = scaler.fit_transform(self.series_original)
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor="train_loss", patience=10, mode="min", verbose=True
        )
        
        # NBEATSx Modell
        nbeatsx_model = NBEATSModel(
            input_chunk_length=336,  # 14 Tage
            output_chunk_length=168, # 1 Woche (anstatt 48h)
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
        
        print("NBEATSx Training...")
        nbeatsx_model.fit(
            series=train_scaled, 
            past_covariates=train_covariates,
            verbose=True
        )
        
        print("NBEATSx Historical Forecasts...")
        nbeatsx_pred_scaled = nbeatsx_model.historical_forecasts(
            series_scaled,
            past_covariates=covariates,
            start=0.8,
            forecast_horizon=1,  # 1-Schritt Vorhersagen
            stride=1,
            retrain=False,
            verbose=True
        )
        
        nbeatsx_pred = scaler.inverse_transform(nbeatsx_pred_scaled)
        
        return self.evaluate_model(nbeatsx_pred, "NBEATSx")
    
    def run_all_models(self):
        """Führt alle vier Modelle aus"""
        print("=" * 80)
        print("    7-DAY MA vs SEASONAL NAIVE vs NBEATS vs NBEATSx")
        print("                  (1 Woche Vorhersagehorizont)")
        print("=" * 80)
        
        self.load_and_prepare_data()
        
        # Alle Modelle ausführen
        self.rolling_7day_moving_average()
        self.seasonal_naive_forecast()
        self.nbeats_forecast()
        self.nbeatsx_forecast()
        
        return self.results
    
    def print_comparison(self):
        """Druckt Vergleichstabelle"""
        print("\n" + "=" * 80)
        print("                    MODELLVERGLEICH ERGEBNISSE")
        print("=" * 80)
        print(f"{'Model':<25} {'MAE':<12} {'MAE%':<12} {'RMSE':<12} {'MAPE':<12}")
        print("-" * 80)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['mae_percent'])
        
        for model_name, result in sorted_results:
            print(f"{model_name:<25} {result['mae']:<12.4f} {result['mae_percent']:<12.2f} "
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
            print(f"NBEATSx vs NBEATS: {improvement:.2f} Prozentpunkte")
        
        if "7Day_Moving_Average" in self.results:
            ma_mae = self.results["7Day_Moving_Average"]["mae_percent"]
            if best_model != "7Day_Moving_Average":
                improvement = ma_mae - best_mae
                print(f"Bestes Modell vs 7-Day MA: {improvement:.2f} Prozentpunkte Verbesserung")
        
        print("=" * 80)
    
    def plot_comprehensive_comparison(self):
        """Erstellt separate Plots für bessere Lesbarkeit"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Plot 1: Gesamtübersicht
        self.plot_overview(timestamp)
        
        # Plot 2: Erste 7 Tage Detail
        self.plot_first_week_detail(timestamp)
        
        # Plot 3: Letzte 7 Tage Detail
        self.plot_last_week_detail(timestamp)
        
        # Plot 4: Performance Balkendiagramm
        self.plot_performance_comparison(timestamp)
    
    def plot_overview(self, timestamp):
        """Plot 1: Gesamtübersicht"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        context_length = min(336, len(self.train_series))
        train_context = self.train_series[-context_length:]
        
        train_context.plot(label="Training (letzte 14 Tage)", ax=ax, linewidth=1.5, alpha=0.7, color='gray')
        self.test_series.plot(label="Echte Werte", ax=ax, linewidth=3, color='black')
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, result) in enumerate(self.results.items()):
            result['predictions'].plot(
                label=f"{model_name} (MAE%: {result['mae_percent']:.2f}%)", 
                ax=ax, linewidth=2.5, linestyle='--', color=colors[i]
            )
        
        # Train/Test Split markieren
        split_point = len(self.train_series)
        split_time = self.series_original.time_index[split_point]
        ax.axvline(x=split_time, color='green', linestyle='-', linewidth=3, alpha=0.8, label='Train/Test Split')
        
        ax.set_title("Modellvergleich: 7-Day Moving Average vs NBEATS vs NBEATSx", fontsize=18, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Energieverbrauch", fontsize=14)
        ax.set_xlabel("Zeit", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"01_Overview_Comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_first_week_detail(self, timestamp):
        """Plot 2: Erste 7 Tage Detail"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        detail_hours = min(168, self.forecast_horizon)  # Erste Woche
        
        # Etwas Training-Kontext für bessere Orientierung
        context = self.train_series[-48:]  # Letzte 2 Tage Training
        context.plot(label="Training (letzte 2 Tage)", ax=ax, linewidth=2, alpha=0.7, color='lightblue')
        
        self.test_series[:detail_hours].plot(label="Echte Werte", ax=ax, linewidth=3, color='black')
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, result) in enumerate(self.results.items()):
            result['predictions'][:detail_hours].plot(
                label=f"{model_name} (MAE%: {result['mae_percent']:.2f}%)", 
                ax=ax, linewidth=2.5, linestyle='--', color=colors[i]
            )
        
        # Tägliche Markierungen
        for day in range(1, 8):
            if day * 24 < detail_hours:
                day_time = self.test_series.time_index[day * 24]
                ax.axvline(x=day_time, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title("Detailansicht: Erste 7 Tage der Vorhersage", fontsize=18, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Energieverbrauch", fontsize=14)
        ax.set_xlabel("Zeit", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"02_First_Week_Detail_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_last_week_detail(self, timestamp):
        """Plot 3: Letzte 7 Tage Detail"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        if self.forecast_horizon >= 168:
            start_idx = -168
            plot_title = "Detailansicht: Letzte 7 Tage der Vorhersage"
        else:
            start_idx = 0
            plot_title = f"Detailansicht: Gesamte Vorhersageperiode ({self.forecast_horizon//24:.1f} Tage)"
        
        self.test_series[start_idx:].plot(label="Echte Werte", ax=ax, linewidth=3, color='black')
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, result) in enumerate(self.results.items()):
            result['predictions'][start_idx:].plot(
                label=f"{model_name} (MAE%: {result['mae_percent']:.2f}%)", 
                ax=ax, linewidth=2.5, linestyle='--', color=colors[i]
            )
        
        ax.set_title(plot_title, fontsize=18, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Energieverbrauch", fontsize=14)
        ax.set_xlabel("Zeit", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"03_Last_Period_Detail_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, timestamp):
        """Plot 4: Performance Balkendiagramm"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        models = list(self.results.keys())
        mae_percents = [self.results[model]['mae_percent'] for model in models]
        
        colors = ['red', 'blue', 'green']
        bars = ax.bar(models, mae_percents, color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=2)
        
        # Werte auf Balken anzeigen
        for bar, value in zip(bars, mae_percents):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Bestes Modell hervorheben
        best_idx = mae_percents.index(min(mae_percents))
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(4)
        
        # Zusätzliche Metriken als Text
        results_text = []
        for model_name, result in self.results.items():
            results_text.append(f"{model_name}:")
            results_text.append(f"  MAE%: {result['mae_percent']:.2f}%")
            results_text.append(f"  RMSE: {result['rmse']:.4f}")
            results_text.append(f"  MAPE: {result['mape']:.2f}%")
            results_text.append("")
        
        ax.text(1.02, 0.98, '\n'.join(results_text), transform=ax.transAxes, 
               verticalalignment='top', fontsize=11, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_ylabel('MAE% (Mittlerer Absoluter Fehler in %)', fontsize=14)
        ax.set_title('Performance Vergleich: MAE%', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(mae_percents) * 1.2)
        
        plt.tight_layout()
        plt.savefig(f"04_Performance_Comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self):
        """Erstellt separate Fehleranalyse-Plots"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Plot 5: Fehler über Zeit
        self.plot_error_over_time(timestamp)
        
        # Plot 6: Scatter Plots
        self.plot_scatter_analysis(timestamp)
        
        # Plot 7: Fehlerstatistiken
        self.plot_error_statistics(timestamp)
    
    def plot_error_over_time(self, timestamp):
        """Plot 5: Fehler über Zeit für alle vier Modelle"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (model_name, result) in enumerate(self.results.items()):
            ax = axes[i]
            
            # Verwende die bereits angepassten Vorhersagen aus results
            pred_values = result['predictions'].values().flatten()
            # Entsprechend angepasste Test-Daten
            actual_values = self.test_series[:len(pred_values)].values().flatten()
            
            errors = actual_values - pred_values
            
            ax.plot(errors, color=colors[i], linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
            ax.fill_between(range(len(errors)), errors, 0, alpha=0.3, color=colors[i])
            
            # Tagesmarkierungen
            for day in range(1, len(errors)//24 + 1):
                if day * 24 < len(errors):
                    ax.axvline(x=day*24, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f"{model_name}\nMAE%: {result['mae_percent']:.2f}%", fontsize=14, fontweight='bold')
            ax.set_xlabel("Stunden", fontsize=11)
            ax.set_ylabel("Vorhersagefehler", fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Statistiken
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(np.abs(errors))
            ax.text(0.02, 0.98, f"μ: {mean_error:.3f}\nσ: {std_error:.3f}\nMax: {max_error:.3f}", 
                   transform=ax.transAxes, verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        plt.suptitle("Fehleranalyse über Zeit: 7-Day MA vs Seasonal Naive vs NBEATS vs NBEATSx", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"05_Error_Over_Time_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scatter_analysis(self, timestamp):
        """Plot 6: Scatter Plots (Ist vs. Vorhersage)"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (model_name, result) in enumerate(self.results.items()):
            ax = axes[i]
            
            # Verwende angepasste Längen
            pred_values = result['predictions'].values().flatten()
            actual_values = self.test_series[:len(pred_values)].values().flatten()
            
            ax.scatter(actual_values, pred_values, alpha=0.6, color=colors[i], s=15, edgecolors='black', linewidth=0.3)
            
            # Perfekte Vorhersage Linie
            min_val = min(np.min(actual_values), np.min(pred_values))
            max_val = max(np.max(actual_values), np.max(pred_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='Perfekt')
            
            ax.set_title(f"{model_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Echte Werte", fontsize=11)
            ax.set_ylabel("Vorhersagewerte", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Korrelation und R²
            correlation = np.corrcoef(actual_values, pred_values)[0, 1]
            r_squared = correlation ** 2
            
            ax.text(0.05, 0.95, f'R: {correlation:.3f}\nR²: {r_squared:.3f}', 
                   transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        plt.suptitle("Scatter Plot Analyse: Echte vs. Vorhergesagte Werte", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"06_Scatter_Analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_statistics(self, timestamp):
        """Plot 7: Fehlerstatistiken Vergleich"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(self.results.keys())
        colors = ['red', 'blue', 'green']
        
        # Alle Metriken sammeln
        mae_values = [self.results[model]['mae'] for model in models]
        mae_percents = [self.results[model]['mae_percent'] for model in models]
        rmse_values = [self.results[model]['rmse'] for model in models]
        mape_values = [self.results[model]['mape'] for model in models]
        
        # Plot 1: MAE
        bars1 = ax1.bar(models, mae_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('MAE')
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars1, mae_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: MAE%
        bars2 = ax2.bar(models, mae_percents, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Mean Absolute Error % (MAE%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE%')
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars2, mae_percents):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_percents)*0.01,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: RMSE
        bars3 = ax3.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('RMSE')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars3, rmse_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: MAPE
        bars4 = ax4.bar(models, mape_values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('MAPE%')
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars4, mape_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_values)*0.01,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Beste Modelle in jeder Kategorie hervorheben
        best_mae_idx = mae_values.index(min(mae_values))
        best_mae_percent_idx = mae_percents.index(min(mae_percents))
        best_rmse_idx = rmse_values.index(min(rmse_values))
        best_mape_idx = mape_values.index(min(mape_values))
        
        bars1[best_mae_idx].set_color('gold')
        bars2[best_mae_percent_idx].set_color('gold')
        bars3[best_rmse_idx].set_color('gold')
        bars4[best_mape_idx].set_color('gold')
        
        plt.suptitle("Vergleich aller Fehlermetriken", fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"07_Error_Statistics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_comparison(self):
        """Führt die komplette Analyse durch"""
        # Alle Modelle ausführen
        self.run_all_models()
        
        # Ergebnisse anzeigen
        self.print_comparison()
        
        # Visualisierungen erstellen
        print("\nErstelle Visualisierungen...")
        self.plot_comprehensive_comparison()
        self.plot_error_analysis()
        
        print("\nVergleich abgeschlossen!")
        return self.results

if __name__ == "__main__":
    comparison = FourModelComparison(
        filepath="id439/imputed_meter_readings_439_CPI_mehr_1.csv"
    )
    
    results = comparison.run_complete_comparison()
    
    print(f"\nFazit:")
    best_model = min(results.items(), key=lambda x: x[1]['mae_percent'])
    print(f"Bestes Modell: {best_model[0]}")
    print(f"Performance: MAE% = {best_model[1]['mae_percent']:.2f}%")
    print(f"Vorhersagehorizont: 1 Woche (168 Stunden)")