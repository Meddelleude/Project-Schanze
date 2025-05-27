import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mape, mae, rmse
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class SmartMeterForecaster:
    def __init__(self, input_chunk_length=24, output_chunk_length=12):
        """
        NBEATSx Model für Smartmeter Datenvorhersage
        
        Args:
            input_chunk_length: Anzahl historischer Zeitschritte als Input
            output_chunk_length: Anzahl vorherzusagender Zeitschritte
        """
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model = None
        self.scaler = None
        self.series = None
        
    def load_smartmeter_data(self, filepath):
        """
        Laden der Smartmeter Daten aus CSV-Datei
        
        Args:
            filepath: Pfad zur CSV-Datei mit 'timestamp' und 'meter_reading' Spalten
        """
        # CSV-Datei laden
        df = pd.read_csv(filepath)
        
        # Überprüfen ob erforderliche Spalten vorhanden sind
        if 'meter_reading' not in df.columns:
            raise ValueError("Spalte 'meter_reading' nicht in der CSV-Datei gefunden")
        
        # Timestamp-Spalte identifizieren (verschiedene mögliche Namen)
        timestamp_cols = ['timestamp', 'datetime', 'date', 'time', 'Date', 'Time']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            # Falls keine Timestamp-Spalte gefunden, Index als Datum verwenden
            print("Keine Timestamp-Spalte gefunden. Erstelle Zeitindex...")
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
            timestamp_col = 'timestamp'
        else:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Daten nach Timestamp sortieren
        df = df.sort_values(by=timestamp_col).reset_index(drop=True)
        
        # Fehlende Werte behandeln
        if df['meter_reading'].isnull().any():
            print(f"Fehlende Werte gefunden: {df['meter_reading'].isnull().sum()}")
            df['meter_reading'] = df['meter_reading'].interpolate(method='linear')
        
        # TimeSeries Objekt erstellen
        self.series = TimeSeries.from_dataframe(
            df, 
            time_col=timestamp_col, 
            value_cols='meter_reading'
        )
        
        print(f"Daten geladen: {len(self.series)} Datenpunkte")
        print(f"Zeitraum: {self.series.start_time()} bis {self.series.end_time()}")
        print(f"Durchschnittlicher Verbrauch: {df['meter_reading'].mean():.4f}")
        print(f"Min/Max Verbrauch: {df['meter_reading'].min():.4f} / {df['meter_reading'].max():.4f}")
        
        return self.series
    
    def prepare_covariates(self):
        """
        Erstelle zusätzliche Features (Kovariaten) für NBEATSx
        """
        # Zeitbasierte Features
        covariates = datetime_attribute_timeseries(
            self.series,
            attribute="hour",
            one_hot=False
        )
        
        # Wochentag hinzufügen
        day_of_week = datetime_attribute_timeseries(
            self.series,
            attribute="dayofweek",
            one_hot=False
        )
        
        # Monat hinzufügen
        month = datetime_attribute_timeseries(
            self.series,
            attribute="month",
            one_hot=False
        )
        
        # Kombiniere alle Kovariaten
        covariates = covariates.stack(day_of_week).stack(month)
        
        return covariates
    
    def preprocess_data(self):
        """
        Datenvorverarbeitung und Skalierung
        """
        # Skalierung der Hauptzeitreihe
        self.scaler = Scaler()
        self.series_scaled = self.scaler.fit_transform(self.series)
        
        # Kovariaten vorbereiten
        self.covariates = self.prepare_covariates()
        
        # Train/Test Split
        split_point = int(0.8 * len(self.series_scaled))
        self.train_series = self.series_scaled[:split_point]
        self.test_series = self.series_scaled[split_point:]
        
        # Kovariaten auch splitten
        self.train_covariates = self.covariates[:split_point + self.output_chunk_length]
        self.test_covariates = self.covariates[split_point:]
        
        print(f"Training-Daten: {len(self.train_series)} Punkte")
        print(f"Test-Daten: {len(self.test_series)} Punkte")
    
    def build_model(self):
        """
        NBEATSx Modell erstellen und konfigurieren
        """
        self.model = NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            num_stacks=3,
            num_blocks=3,
            num_layers=4,
            layer_widths=256,
            expansion_coefficient_dim=5,
            trend_polynomial_degree=3,
            dropout=0.1,
            activation="ReLU",
            n_epochs=50,
            batch_size=32,
            optimizer_kwargs={'lr': 1e-3},
            model_name="NBEATSx_SmartMeter",
            save_checkpoints=True,
            force_reset=True,
            pl_trainer_kwargs={
                "enable_progress_bar": True,
                "enable_model_summary": True,
                "callbacks": [
                    # Early Stopping auf train_loss statt val_loss
                    EarlyStopping(
                        monitor="train_loss",
                        patience=10,
                        min_delta=0.0001,
                        mode="min",
                        verbose=True
                    )
                ]
            }
        )
        
        print("NBEATSx Modell mit Early Stopping erstellt")
    
    def train_model(self):
        """
        Modell trainieren
        """
        print("Starte Training...")
        
        self.model.fit(
            series=self.train_series,
            past_covariates=self.train_covariates,
            verbose=True
        )
        
        print("Training abgeschlossen!")
    
    def make_predictions(self, n_steps=None):
        """
        Vorhersagen erstellen
        """
        if n_steps is None:
            n_steps = len(self.test_series)
        
        predictions = self.model.predict(
            n=n_steps,
            series=self.train_series,
            past_covariates=self.covariates
        )
        
        # Rück-Skalierung
        predictions_rescaled = self.scaler.inverse_transform(predictions)
        
        return predictions_rescaled
    
    def evaluate_model(self, predictions):
        """
        Modellbewertung
        """
        # Testdaten rück-skalieren für Vergleich
        test_rescaled = self.scaler.inverse_transform(self.test_series)
        
        # Metriken berechnen
        mape_score = mape(test_rescaled, predictions)
        mae_score = mae(test_rescaled, predictions)
        rmse_score = rmse(test_rescaled, predictions)
        
        # MAE als Prozentsatz berechnen
        mean_actual = test_rescaled.values().mean()
        mae_percentage = (mae_score / mean_actual) * 100
        
        print("\n=== Modell-Evaluation ===")
        print(f"MAPE: {mape_score:.2f}%")
        print(f"MAE: {mae_score:.4f}")
        print(f"MAE%: {mae_percentage:.2f}%")
        print(f"RMSE: {rmse_score:.4f}")
        
        return {
            'mape': mape_score,
            'mae': mae_score,
            'mae_percentage': mae_percentage,
            'rmse': rmse_score
        }
    
    def plot_results(self, predictions, n_plot_points=168):  # 1 Woche
        """
        Ergebnisse visualisieren
        """
        # Nur die letzten n_plot_points für bessere Sichtbarkeit
        test_rescaled = self.scaler.inverse_transform(self.test_series)
        
        plt.figure(figsize=(15, 8))
        
        # Trainingsdaten (letzte Woche)
        train_rescaled = self.scaler.inverse_transform(self.train_series)
        train_plot = train_rescaled[-n_plot_points:]
        train_plot.plot(label="Training Data", color='blue', alpha=0.7)
        
        # Testdaten
        test_plot = test_rescaled[:n_plot_points]
        test_plot.plot(label="Actual", color='green', linewidth=2)
        
        # Vorhersagen
        pred_plot = predictions[:n_plot_points]
        pred_plot.plot(label="Predicted", color='red', linewidth=2, linestyle='--')
        
        plt.title('Smartmeter Energieverbrauch Vorhersage', fontsize=16)
        plt.xlabel('Zeit')
        plt.ylabel('Energieverbrauch (kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self, filepath):
        """
        Komplette Pipeline ausführen
        
        Args:
            filepath: Pfad zur CSV-Datei mit Smartmeter-Daten
        """
        print("=== NBEATSx Smartmeter Forecasting Pipeline ===\n")
        
        # 1. Daten laden
        self.load_smartmeter_data(filepath)
        
        # 2. Daten vorverarbeiten
        self.preprocess_data()
        
        # 3. Modell erstellen
        self.build_model()
        
        # 4. Modell trainieren
        self.train_model()
        
        # 5. Vorhersagen erstellen
        predictions = self.make_predictions()
        
        # 6. Modell evaluieren
        metrics = self.evaluate_model(predictions)
        
        # 7. Ergebnisse visualisieren
        self.plot_results(predictions)
        
        return predictions, metrics

# Verwendung
if __name__ == "__main__":
    # Forecaster initialisieren
    forecaster = SmartMeterForecaster(
        input_chunk_length=336,  # 336 Stunden Input (14 Tage)
        output_chunk_length=48   # 48 Stunden Vorhersage (2 Tage)
    )
    
    # Pipeline mit eigenen Daten ausführen
    csv_path = "id118/imputed_meter_readings_118_CPI.csv"  # Pfad zu Ihrer CSV-Datei
    predictions, metrics = forecaster.run_complete_pipeline(csv_path)
    
    print("\nPipeline erfolgreich abgeschlossen!")
    print(f"Modell-Performance: MAPE = {metrics['mape']:.2f}%, MAE% = {metrics['mae_percentage']:.2f}%")