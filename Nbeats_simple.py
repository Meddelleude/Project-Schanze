import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse, mape
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NBEATSForecaster:
    def __init__(self, filepath, input_chunk_length=336, output_chunk_length=48):
        self.filepath = filepath
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model = None
        self.scaler = None
        self.series = None
        
    def load_and_prepare_data(self):
        print("Daten laden")
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        self.series_original = TimeSeries.from_dataframe(df, "timestamp", "meter_reading")
        
        self.scaler = Scaler()
        self.series_scaled = self.scaler.fit_transform(self.series_original)
        
        self.train_series = self.series_scaled[:int(0.8 * len(self.series_scaled))]
        self.test_series = self.series_scaled[int(0.8 * len(self.series_scaled)):]
        
        self.test_series_unscaled = self.scaler.inverse_transform(self.test_series)
        
        print(f"Training: {len(self.train_series)} Punkte")
        print(f"Test: {len(self.test_series)} Punkte")
        print(f"Test repräsentiert: {len(self.test_series)/len(self.series_scaled)*100:.1f}% der Daten")
        
    def train_nbeats(self):
        print("NBEATS Training")
        
        early_stopping = EarlyStopping(
            monitor="train_loss", patience=10, mode="min", verbose=True
        )
        
        self.model = NBEATSModel(
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
                "callbacks": [early_stopping],
                "logger": False,
                "enable_checkpointing": False
            }
        )
        
        self.model.fit(series=self.train_series, verbose=True)
        print("NBEATS Training abgeschlossen")
        
    def predict_and_evaluate(self):
        print("Vorhersagen und Evaluation")
        
        nbeats_pred_scaled = self.model.predict(
            n=len(self.test_series),
            series=self.train_series
        )
        nbeats_pred = self.scaler.inverse_transform(nbeats_pred_scaled)
        
        mean_actual = self.test_series_unscaled.values().mean()
        
        nbeats_mae = mae(self.test_series_unscaled, nbeats_pred)
        nbeats_mae_percent = (nbeats_mae / mean_actual) * 100
        nbeats_rmse = rmse(self.test_series_unscaled, nbeats_pred)
        nbeats_mape = mape(self.test_series_unscaled, nbeats_pred)
        
        self.results = {
            'predictions': nbeats_pred,
            'mae': nbeats_mae,
            'mae_percent': nbeats_mae_percent,
            'rmse': nbeats_rmse,
            'mape': nbeats_mape
        }
        
    def print_results(self):
        print("=" * 50)
        print("           NBEATS ERGEBNISSE")
        print("=" * 50)
        
        print(f"MAE:    {self.results['mae']:.4f}")
        print(f"MAE%:   {self.results['mae_percent']:.2f}%")
        print(f"RMSE:   {self.results['rmse']:.4f}")
        print(f"MAPE:   {self.results['mape']:.2f}%")
        print("=" * 50)
        
    def plot_results(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        self.series_original.plot(label="Echte Werte", ax=ax1, linewidth=0.8, alpha=0.7)
        self.results['predictions'].plot(
            label="NBEATS Vorhersage", ax=ax1, linewidth=1.5, color='red'
        )
        
        test_start = self.series_original.time_index[int(0.8 * len(self.series_original))]
        ax1.axvline(x=test_start, color='green', linestyle='--', alpha=0.7, label='Test Start')
        
        ax1.legend()
        ax1.set_title("NBEATS Gesamtübersicht")
        ax1.set_ylabel("Energieverbrauch")
        ax1.grid(True, alpha=0.3)
        
        self.test_series_unscaled.plot(
            label="Echte Werte (Test)", ax=ax2, linewidth=2, color='black'
        )
        self.results['predictions'].plot(
            label=f"NBEATS (MAE%: {self.results['mae_percent']:.2f}%)", 
            ax=ax2, linewidth=1.5, color='red'
        )
        
        ax2.legend()
        ax2.set_title("NBEATS Detailansicht: Vorhersagebereich (letzte 20%)")
        ax2.set_ylabel("Energieverbrauch")
        ax2.set_xlabel("Zeit")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"NBEATS_Results_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_full_forecast(self):
        print("Starte NBEATS Vorhersage...")
        print(f"Vorhersage der letzten 20% der Daten")
        print(f"Input: {self.input_chunk_length}h, Output: {self.output_chunk_length}h")
        
        self.load_and_prepare_data()
        self.train_nbeats()
        self.predict_and_evaluate()
        self.print_results()
        self.plot_results()
        
        return self.results

if __name__ == "__main__":
    forecaster = NBEATSForecaster(
        filepath="id118/imputed_meter_readings_118_CPI.csv",
        input_chunk_length=336,
        output_chunk_length=48
    )
    
    results = forecaster.run_full_forecast()
    
    print(f"NBEATS Vorhersage abgeschlossen!")
    print(f"MAE%: {results['mae_percent']:.2f}%")