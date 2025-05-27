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

class NBEATSComparison:
    def __init__(self, filepath, input_chunk_length=336, output_chunk_length=48):
        self.filepath = filepath
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.results = {}
        
    def load_and_prepare_data(self):
        """Daten laden und vorbereiten"""
        print("=== Daten laden ===")
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        self.series_original = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
        
        # Skalierung
        self.scaler = Scaler()
        self.series_scaled = self.scaler.fit_transform(self.series_original)
        
        # 80/20 Split
        self.train_series = self.series_scaled[:int(0.8 * len(self.series_scaled))]
        self.test_series = self.series_scaled[int(0.8 * len(self.series_scaled)):]
        
        # Test-Daten rück-skalieren für Evaluation
        self.test_series_unscaled = self.scaler.inverse_transform(self.test_series)
        
        print(f"Training: {len(self.train_series)} Punkte")
        print(f"Test: {len(self.test_series)} Punkte")
        print(f"Test repräsentiert: {len(self.test_series)/len(self.series_scaled)*100:.1f}% der Daten")
        
    def prepare_covariates_for_nbeatsx(self):
        """Kovariaten für NBEATSx vorbereiten"""
        print("Kovariaten für NBEATSx vorbereiten...")
        
        # Zeitbasierte Features für gesamte Serie
        hour_cov = datetime_attribute_timeseries(
            self.series_original, attribute="hour", one_hot=False
        )
        day_cov = datetime_attribute_timeseries(
            self.series_original, attribute="dayofweek", one_hot=False
        )
        month_cov = datetime_attribute_timeseries(
            self.series_original, attribute="month", one_hot=False
        )
        
        # Kombinieren
        self.covariates = hour_cov.stack(day_cov).stack(month_cov)
        
        # Kovariaten für Training (muss über Testbereich hinausgehen für Vorhersagen)
        train_end = len(self.train_series)
        self.train_covariates = self.covariates[:train_end + len(self.test_series)]
        
    def train_nbeats(self):
        """NBEATS (ohne Kovariaten) trainieren"""
        print("\n=== NBEATS Training ===")
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor="train_loss", patience=10, mode="min", verbose=True
        )
        
        # NBEATS Modell
        self.nbeats_model = NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
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
                "callbacks": [early_stopping]
            }
        )
        
        # Training
        self.nbeats_model.fit(series=self.train_series, verbose=True)
        print("NBEATS Training abgeschlossen!")
        
    def train_nbeatsx(self):
        """NBEATSx (mit Kovariaten) trainieren"""
        print("\n=== NBEATSx Training ===")
        
        # Kovariaten vorbereiten
        self.prepare_covariates_for_nbeatsx()
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor="train_loss", patience=10, mode="min", verbose=True
        )
        
        # NBEATSx Modell
        self.nbeatsx_model = NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
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
                "callbacks": [early_stopping]
            }
        )
        
        # Training mit Kovariaten
        self.nbeatsx_model.fit(
            series=self.train_series, 
            past_covariates=self.train_covariates,
            verbose=True
        )
        print("NBEATSx Training abgeschlossen!")
        
    def predict_and_evaluate(self):
        """Vorhersagen erstellen und bewerten"""
        print("\n=== Vorhersagen und Evaluation ===")
        
        # NBEATS Vorhersage
        print("NBEATS Vorhersage...")
        nbeats_pred_scaled = self.nbeats_model.predict(
            n=len(self.test_series),
            series=self.train_series
        )
        nbeats_pred = self.scaler.inverse_transform(nbeats_pred_scaled)
        
        # NBEATSx Vorhersage
        print("NBEATSx Vorhersage...")
        nbeatsx_pred_scaled = self.nbeatsx_model.predict(
            n=len(self.test_series),
            series=self.train_series,
            past_covariates=self.covariates
        )
        nbeatsx_pred = self.scaler.inverse_transform(nbeatsx_pred_scaled)
        
        # Metriken berechnen
        mean_actual = self.test_series_unscaled.values().mean()
        
        # NBEATS Metriken
        nbeats_mae = mae(self.test_series_unscaled, nbeats_pred)
        nbeats_mae_percent = (nbeats_mae / mean_actual) * 100
        nbeats_rmse = rmse(self.test_series_unscaled, nbeats_pred)
        nbeats_mape = mape(self.test_series_unscaled, nbeats_pred)
        
        # NBEATSx Metriken
        nbeatsx_mae = mae(self.test_series_unscaled, nbeatsx_pred)
        nbeatsx_mae_percent = (nbeatsx_mae / mean_actual) * 100
        nbeatsx_rmse = rmse(self.test_series_unscaled, nbeatsx_pred)
        nbeatsx_mape = mape(self.test_series_unscaled, nbeatsx_pred)
        
        # Ergebnisse speichern
        self.results = {
            'nbeats': {
                'predictions': nbeats_pred,
                'mae': nbeats_mae,
                'mae_percent': nbeats_mae_percent,
                'rmse': nbeats_rmse,
                'mape': nbeats_mape
            },
            'nbeatsx': {
                'predictions': nbeatsx_pred,
                'mae': nbeatsx_mae,
                'mae_percent': nbeatsx_mae_percent,
                'rmse': nbeatsx_rmse,
                'mape': nbeatsx_mape
            }
        }
        
    def print_comparison(self):
        """Vergleichsergebnisse ausgeben"""
        print("\n" + "="*60)
        print("           MODELLVERGLEICH - ERGEBNISSE")
        print("="*60)
        
        print(f"\n📊 NBEATS (ohne Kovariaten):")
        print(f"   MAE:    {self.results['nbeats']['mae']:.4f}")
        print(f"   MAE%:   {self.results['nbeats']['mae_percent']:.2f}%")
        print(f"   RMSE:   {self.results['nbeats']['rmse']:.4f}")
        print(f"   MAPE:   {self.results['nbeats']['mape']:.2f}%")
        
        print(f"\n🚀 NBEATSx (mit Zeitkovariaten):")
        print(f"   MAE:    {self.results['nbeatsx']['mae']:.4f}")
        print(f"   MAE%:   {self.results['nbeatsx']['mae_percent']:.2f}%")
        print(f"   RMSE:   {self.results['nbeatsx']['rmse']:.4f}")
        print(f"   MAPE:   {self.results['nbeatsx']['mape']:.2f}%")
        
        # Vergleich
        mae_diff = self.results['nbeats']['mae_percent'] - self.results['nbeatsx']['mae_percent']
        
        print(f"\n🏆 GEWINNER basierend auf MAE%:")
        if mae_diff > 0:
            print(f"   NBEATSx ist BESSER um {mae_diff:.2f} Prozentpunkte!")
            winner = "NBEATSx"
        elif mae_diff < 0:
            print(f"   NBEATS ist BESSER um {abs(mae_diff):.2f} Prozentpunkte!")
            winner = "NBEATS"
        else:
            print("   UNENTSCHIEDEN!")
            winner = "Tie"
            
        # Verbesserung in Prozent
        if mae_diff > 0:
            improvement = (mae_diff / self.results['nbeats']['mae_percent']) * 100
            print(f"   NBEATSx zeigt {improvement:.1f}% Verbesserung gegenüber NBEATS")
        elif mae_diff < 0:
            improvement = (abs(mae_diff) / self.results['nbeatsx']['mae_percent']) * 100
            print(f"   NBEATS zeigt {improvement:.1f}% Verbesserung gegenüber NBEATSx")
            
        print("="*60)
        
        return winner, mae_diff
        
    def plot_comparison(self):
        """Vergleichsplot erstellen"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: Gesamtübersicht
        self.series_original.plot(label="Echte Werte", ax=ax1, linewidth=0.8, alpha=0.7)
        self.results['nbeats']['predictions'].plot(
            label="NBEATS Vorhersage", ax=ax1, linewidth=1.5, color='red'
        )
        self.results['nbeatsx']['predictions'].plot(
            label="NBEATSx Vorhersage", ax=ax1, linewidth=1.5, color='blue'
        )
        
        # Testbereich markieren
        test_start = self.series_original.time_index[int(0.8 * len(self.series_original))]
        ax1.axvline(x=test_start, color='green', linestyle='--', alpha=0.7, label='Test Start')
        
        ax1.legend()
        ax1.set_title("Gesamtübersicht: NBEATS vs NBEATSx")
        ax1.set_ylabel("Energieverbrauch")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Nur Testbereich (Zoom)
        self.test_series_unscaled.plot(
            label="Echte Werte (Test)", ax=ax2, linewidth=2, color='black'
        )
        self.results['nbeats']['predictions'].plot(
            label=f"NBEATS (MAE%: {self.results['nbeats']['mae_percent']:.2f}%)", 
            ax=ax2, linewidth=1.5, color='red'
        )
        self.results['nbeatsx']['predictions'].plot(
            label=f"NBEATSx (MAE%: {self.results['nbeatsx']['mae_percent']:.2f}%)", 
            ax=ax2, linewidth=1.5, color='blue'
        )
        
        ax2.legend()
        ax2.set_title("Detailansicht: Vorhersagebereich (letzte 20%)")
        ax2.set_ylabel("Energieverbrauch")
        ax2.set_xlabel("Zeit")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"NBEATS_vs_NBEATSx_Comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Metriken-Vergleichsplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models = ['NBEATS', 'NBEATSx']
        mae_percents = [
            self.results['nbeats']['mae_percent'],
            self.results['nbeatsx']['mae_percent']
        ]
        colors = ['red', 'blue']
        
        bars = ax.bar(models, mae_percents, color=colors, alpha=0.7, edgecolor='black')
        
        # Werte auf Balken anzeigen
        for bar, value in zip(bars, mae_percents):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('MAE% (Mittlerer Absoluter Fehler in %)')
        ax.set_title('Modellvergleich: MAE% Performance')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Gewinner hervorheben
        winner_idx = 0 if mae_percents[0] < mae_percents[1] else 1
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(f"MAE_Comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_full_comparison(self):
        """Vollständigen Vergleich durchführen"""
        print("🚀 Starte NBEATS vs NBEATSx Vergleich...")
        print(f"📊 Vorhersage der letzten 20% der Daten")
        print(f"🔍 Vergleichsmetrik: MAE%")
        print(f"⚙️  Input: {self.input_chunk_length}h, Output: {self.output_chunk_length}h")
        
        # 1. Daten vorbereiten
        self.load_and_prepare_data()
        
        # 2. Beide Modelle trainieren
        self.train_nbeats()
        self.train_nbeatsx()
        
        # 3. Vorhersagen und Evaluation
        self.predict_and_evaluate()
        
        # 4. Ergebnisse anzeigen
        winner, improvement = self.print_comparison()
        
        # 5. Plots erstellen
        self.plot_comparison()
        
        return self.results, winner, improvement

# Verwendung
if __name__ == "__main__":
    # Vergleich durchführen
    comparison = NBEATSComparison(
        filepath="id118/imputed_meter_readings_118_CPI.csv",
        input_chunk_length=336,  # 14 Tage
        output_chunk_length=48   # 2 Tage
    )
    
    # Vollständigen Vergleich ausführen
    results, winner, improvement = comparison.run_full_comparison()
    
    print(f"\n🎯 FAZIT:")
    print(f"Der Gewinner ist: {winner}")
    if improvement != 0:
        print(f"Verbesserung: {abs(improvement):.2f} Prozentpunkte bei MAE%")