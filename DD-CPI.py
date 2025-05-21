import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')  

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def single_value_linear_interpolation(energy_ts):
    energy_ts_indexed = energy_ts.set_index('timestamp')
    mask = energy_ts_indexed['anomaly'] == 1
    single_missing = mask & ~(mask.shift(1) & mask | mask & mask.shift(-1))
    interpolated_ts = energy_ts_indexed.copy()
    single_missing_idx = interpolated_ts.index[single_missing]
    
    print(f"Einzelne fehlende Werte gefunden: {len(single_missing_idx)}")
    
    for idx in single_missing_idx:
        prev_idx = interpolated_ts.index[interpolated_ts.index < idx][-1] if any(interpolated_ts.index < idx) else None
        next_idx = interpolated_ts.index[interpolated_ts.index > idx][0] if any(interpolated_ts.index > idx) else None
        
        if prev_idx is not None and next_idx is not None:
            prev_value = interpolated_ts.loc[prev_idx, 'meter_reading']
            next_value = interpolated_ts.loc[next_idx, 'meter_reading']
            time_diff_prev = (idx - prev_idx).total_seconds() / 3600  
            time_diff_total = (next_idx - prev_idx).total_seconds() / 3600
            interpolated_value = prev_value + (next_value - prev_value) * (time_diff_prev / time_diff_total)
            interpolated_value = max(0, interpolated_value)
            interpolated_ts.loc[idx, 'meter_reading'] = interpolated_value
            interpolated_ts.loc[idx, 'anomaly'] = 0  
    return interpolated_ts.reset_index()

def calculate_energy_per_day(energy_ts):
    energy_ts['date'] = energy_ts['timestamp'].dt.date
    daily_energy = energy_ts.groupby('date').agg({
        'meter_reading': 'sum',  
        'anomaly': lambda x: 1 if any(x == 1) else 0,  
        'building_id': 'first' 
    }).reset_index()
    daily_energy['timestamp'] = pd.to_datetime(daily_energy['date'])
    daily_energy.drop('date', axis=1, inplace=True)
    daily_energy.set_index('timestamp', inplace=True)
    
    print(f"Tägliche Energiewerte berechnet: {len(daily_energy)} Tage")
    print(f"Davon Tage mit Anomalien: {daily_energy['anomaly'].sum()}")
    
    return daily_energy

def determine_days_with_missing_values(energy_ts):
    energy_ts['date'] = energy_ts['timestamp'].dt.date
    days_with_missing = energy_ts.groupby('date')['anomaly'].max()
    
    print(f"Tage mit fehlenden Werten ermittelt: {days_with_missing.sum()} von {len(days_with_missing)}")
    
    return days_with_missing.values

def estimate_weekly_pattern_with_prophet(daily_energy, non_complete_days):
    print(f"Estimating weekly pattern with Prophet...")
    print(f"Daily energy shape: {daily_energy.shape}")
    print(f"Non-complete days shape: {non_complete_days.shape}")
    complete_days = ~non_complete_days.astype(bool)
    print(f"Complete days: {sum(complete_days)} von {len(complete_days)}")

    if not any(complete_days):
        print("WARNUNG: Keine vollständigen Tage zum Trainieren des Prophet-Modells gefunden.")
        print("Verwende einen einfachen Wochentag-basierten Fallback-Ansatz.")
        weekdays = [daily_energy.index[i].weekday() for i in range(len(daily_energy))]
        fallback_pattern = []
        for wd in weekdays:
            if wd < 5: 
                fallback_pattern.append(0.1)  
            else:  
                fallback_pattern.append(-0.15) 
        fallback_pattern = np.array(fallback_pattern)
        fallback_pattern = fallback_pattern - np.mean(fallback_pattern)
        
        return fallback_pattern
    prophet_df = daily_energy.reset_index().copy()
    prophet_df = prophet_df[complete_days]
    
    print(f"Prophet DataFrame shape nach Filterung: {prophet_df.shape}")
    prophet_df = prophet_df.rename(columns={'timestamp': 'ds', 'meter_reading': 'y'})
    print("Erste Zeilen des Prophet DataFrame:")
    print(prophet_df.head())
    if len(prophet_df) == 0:
        print("WARNUNG: Nach Filterung keine Daten für Prophet übrig.")
        weekdays = [daily_energy.index[i].weekday() for i in range(len(daily_energy))]
        
        fallback_pattern = []
        for wd in weekdays:
            if wd < 5:  
                fallback_pattern.append(0.1)  
            else:  
                fallback_pattern.append(-0.15) 
        
        fallback_pattern = np.array(fallback_pattern)
        fallback_pattern = fallback_pattern - np.mean(fallback_pattern)
        
        return fallback_pattern
    
    try:
        print("Prophet Daten überprüfen:")
        print(f"Sind NaN-Werte vorhanden? {prophet_df['y'].isna().any()}")
        prophet_df = prophet_df.dropna(subset=['y'])
        
        print(f"Prophet DataFrame shape nach NaN-Entfernung: {prophet_df.shape}")
        
        if len(prophet_df) == 0:
            raise ValueError("Alle Werte sind NaN!")
        model = Prophet(weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df[['ds', 'y']])
        future = pd.DataFrame({'ds': daily_energy.index.tolist()})
        
        print(f"Future DataFrame shape: {future.shape}")
        print("Erste Zeilen des Future DataFrame:")
        print(future.head())
        
        forecast = model.predict(future)
        weekly_pattern = forecast['weekly']
        
        print(f"Weekly pattern berechnet mit Länge: {len(weekly_pattern)}")
        
        return weekly_pattern.values
    except Exception as e:
        print(f"Fehler beim Trainieren des Prophet-Modells: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        print("Verwende einen Fallback-Ansatz.")
        weekdays = [daily_energy.index[i].weekday() for i in range(len(daily_energy))]
        
        fallback_pattern = []
        for wd in weekdays:
            if wd < 5: 
                fallback_pattern.append(0.1)  
            else: 
                fallback_pattern.append(-0.15)  
        
        fallback_pattern = np.array(fallback_pattern)
        fallback_pattern = fallback_pattern - np.mean(fallback_pattern)
        
        return fallback_pattern

def estimate_missing_energy_per_day(energy_ts, weekly_pattern):
    days = pd.date_range(
        start=energy_ts['timestamp'].min().normalize(), 
        end=energy_ts['timestamp'].max().normalize(),
        freq='D'
    )
    missing_energy = pd.Series(0, index=days)
    energy_ts['gap_id'] = (energy_ts['anomaly'].diff() == 1).cumsum()
    gaps = energy_ts[energy_ts['anomaly'] == 1].groupby('gap_id')
    
    print(f"Anzahl identifizierter Lücken: {len(gaps)}")
    
    for gap_id, gap_data in gaps:
        gap_start = gap_data['timestamp'].min()
        gap_end = gap_data['timestamp'].max()
        
        print(f"Verarbeite Lücke {gap_id}: {gap_start} bis {gap_end}")
        gap_hours = gap_data['timestamp'].dt.hour.values
        gap_weekdays = gap_data['timestamp'].dt.weekday.values
        similar_hours_data = []
        for hour, weekday in zip(gap_hours, gap_weekdays):
            similar_data = energy_ts[
                (energy_ts['anomaly'] == 0) &  
                (energy_ts['timestamp'].dt.hour == hour) &  
                (energy_ts['timestamp'].dt.weekday == weekday)  
            ]
            
            if len(similar_data) > 0:
                avg_consumption = similar_data['meter_reading'].mean()
            else:
                similar_time = energy_ts[
                    (energy_ts['anomaly'] == 0) &
                    (energy_ts['timestamp'].dt.hour == hour)
                ]
                
                if len(similar_time) > 0:
                    avg_consumption = similar_time['meter_reading'].mean()
                else:
                    avg_consumption = energy_ts[energy_ts['anomaly'] == 0]['meter_reading'].mean()
            
            similar_hours_data.append(avg_consumption)
        estimated_gap_energy = sum(similar_hours_data)
        
        print(f"  Geschätzte Energie während der Lücke: {estimated_gap_energy:.2f}")

        gap_days = pd.date_range(gap_start.normalize(), gap_end.normalize(), freq='D')
        days_in_gap = len(gap_days)
        
        print(f"  Tage in der Lücke: {days_in_gap}")
        missing_per_day = {}
        for day in gap_days:
            day_start = day
            day_end = day + timedelta(days=1)
            missing_in_day = len(gap_data[(gap_data['timestamp'] >= day_start) & 
                                         (gap_data['timestamp'] < day_end)])
            missing_per_day[day] = missing_in_day
        
        total_missing = sum(missing_per_day.values())
        
        print(f"  Fehlende Werte pro Tag: {missing_per_day}")
        print(f"  Gesamtzahl fehlender Werte: {total_missing}")
        for day, count in missing_per_day.items():
            day_energy_share = estimated_gap_energy * (count / total_missing) if total_missing > 0 else 0
            if day in missing_energy.index:
                missing_energy.loc[day] += day_energy_share
        if days_in_gap > 1 and len(weekly_pattern) > 0:
            day_indices = [i % 7 for i in range(len(gap_days))]
            if len(weekly_pattern) >= 7:
                weekly_adjustments = np.array([weekly_pattern[idx] for idx in day_indices])
            else:
                weekly_adjustments = np.zeros(len(gap_days))
                print("  WARNUNG: weekly_pattern zu kurz, verwende Nullen als Anpassung")
            if len(weekly_adjustments) > 0:
                weekly_adjustments = weekly_adjustments - np.mean(weekly_adjustments)
            for i, day in enumerate(gap_days):
                if day in missing_energy.index and i < len(weekly_adjustments):
                    missing_energy.loc[day] += weekly_adjustments[i]
    
    return missing_energy

def compile_list_of_complete_days(timestamps, daily_energy, non_complete_days):
    complete_days = []
    
    for i, (timestamp, is_incomplete) in enumerate(zip(timestamps, non_complete_days)):
        if not is_incomplete and i < len(daily_energy):
            energy = daily_energy.iloc[i]['meter_reading']
            
            if not pd.isna(energy):
                weekday = timestamp.weekday() + 1  
                seasonal_pos = timestamp.timetuple().tm_yday
                
                complete_days.append({
                    'date': timestamp,
                    'energy': energy,
                    'weekday': weekday,
                    'seasonal_pos': seasonal_pos
                })
    
    print(f"Liste vollständiger Tage erstellt: {len(complete_days)} Tage")
    
    return complete_days

def find_day_with_min_dissimilarity(day_with_gaps, complete_days, weights=(5, 1, 10)):

    we, ww, ws = weights  
    gap_day_date = day_with_gaps['date']
    gap_day_energy = day_with_gaps['energy']
    gap_day_weekday = gap_day_date.weekday() + 1  
    gap_day_seasonal = gap_day_date.timetuple().tm_yday 
    min_dissimilarity = float('inf')
    best_matching_day = None
    energy_values = [day['energy'] for day in complete_days]
    e_min, e_max = min(energy_values), max(energy_values)
    e_range = e_max - e_min if e_max > e_min else 1  
    
    for complete_day in complete_days:
        de = abs(gap_day_energy - complete_day['energy']) / e_range
        if gap_day_weekday == complete_day['weekday']:
            dw = 0.0
        elif (gap_day_weekday <= 5 and complete_day['weekday'] <= 5) or \
             (gap_day_weekday >= 6 and complete_day['weekday'] >= 6):
            dw = 0.5  
        else:
            dw = 1.0  
        s = 366  
        seasonal_diff = abs(gap_day_seasonal - complete_day['seasonal_pos'])
        
        if seasonal_diff <= s / 2:
            ds = seasonal_diff / (s / 2)
        else:
            ds = (s - seasonal_diff) / (s / 2)
        dissimilarity = we * de + ww * dw + ws * ds
        
        if dissimilarity < min_dissimilarity:
            min_dissimilarity = dissimilarity
            best_matching_day = complete_day
    
    return best_matching_day

def dd_cpi_imputation(energy_ts, weights=(5, 1, 10)):

    print("Schritt 1: Lineare Interpolation einzelner fehlender Werte...")
    energy_ts_processed = single_value_linear_interpolation(energy_ts)
    print("Schritt 2: Energieverbrauchsschätzung...")
    daily_energy = calculate_energy_per_day(energy_ts_processed)
    non_complete_days = determine_days_with_missing_values(energy_ts_processed)
    weekly_pattern = estimate_weekly_pattern_with_prophet(daily_energy, non_complete_days)
    missing_energy = estimate_missing_energy_per_day(energy_ts_processed, weekly_pattern)
    estimated_energy_per_day = daily_energy.copy()
    for day in missing_energy.index:
        if day in estimated_energy_per_day.index:
            estimated_energy_per_day.loc[day, 'meter_reading'] += missing_energy.loc[day]
    print("Schritt 3: Zusammenstellung der verfügbaren vollständigen Tage...")
    complete_days = compile_list_of_complete_days(
        daily_energy.index, 
        daily_energy, 
        non_complete_days
    )
    
    print("Schritt 4: Vorbereitung für die Imputation...")
    imputed_ts = energy_ts_processed.copy()
    print("Schritt 5: Finde und kopiere die besten passenden Tage...")
    days_with_gaps = []
    for day_idx, is_incomplete in enumerate(non_complete_days):
        if is_incomplete and day_idx < len(daily_energy.index):
            day_date = daily_energy.index[day_idx]
            days_with_gaps.append({
                'date': day_date,
                'energy': estimated_energy_per_day.loc[day_date, 'meter_reading'] if day_date in estimated_energy_per_day.index else 0
            })
    for day_with_gaps in days_with_gaps:
        best_matching_day = find_day_with_min_dissimilarity(day_with_gaps, complete_days, weights)
        
        if best_matching_day is None:
            print(f"WARNUNG: Kein passender Tag gefunden für {day_with_gaps['date']}")
            continue
        
        print(f"Bester passender Tag für {day_with_gaps['date']}: {best_matching_day['date']}")
        match_day_start = best_matching_day['date']
        match_day_end = match_day_start + timedelta(days=1)
        
        matching_day_data = energy_ts_processed[
            (energy_ts_processed['timestamp'] >= match_day_start) & 
            (energy_ts_processed['timestamp'] < match_day_end) &
            (energy_ts_processed['anomaly'] == 0)
        ]
        hour_to_consumption = {}
        for _, row in matching_day_data.iterrows():
            hour_key = row['timestamp'].hour
            hour_to_consumption[hour_key] = row['meter_reading']
        gap_day_start = day_with_gaps['date']
        gap_day_end = gap_day_start + timedelta(days=1)
        
        gap_day_anomalies = imputed_ts[
            (imputed_ts['timestamp'] >= gap_day_start) & 
            (imputed_ts['timestamp'] < gap_day_end) &
            (imputed_ts['anomaly'] == 1)
        ]
        
        for _, row in gap_day_anomalies.iterrows():
            ts_hour = row['timestamp'].hour
            if ts_hour in hour_to_consumption:
                imputed_value = hour_to_consumption[ts_hour]
                imputed_ts.loc[imputed_ts['timestamp'] == row['timestamp'], 'meter_reading'] = imputed_value
                imputed_ts.loc[imputed_ts['timestamp'] == row['timestamp'], 'anomaly'] = 0  
    print("Schritt 6: Skaliere die imputierten Werte...")
    energy_ts['gap_id'] = (energy_ts['anomaly'].diff() == 1).cumsum()
    gaps = energy_ts[energy_ts['anomaly'] == 1].groupby('gap_id')
    
    for gap_id, gap_data in gaps:
        gap_start = gap_data['timestamp'].min()
        gap_end = gap_data['timestamp'].max()
        imputed_values = imputed_ts[
            (imputed_ts['timestamp'] >= gap_start) & 
            (imputed_ts['timestamp'] <= gap_end) &
            (imputed_ts['anomaly'] == 0)  
        ]['meter_reading']
        
        if len(imputed_values) == 0:
            print(f"  WARNUNG: Keine imputierten Werte gefunden für Lücke {gap_id}")
            continue
        imputed_sum = imputed_values.sum()
        gap_hours = gap_data['timestamp'].dt.hour.values
        gap_weekdays = gap_data['timestamp'].dt.weekday.values
        
        estimated_values = []
        for hour, weekday in zip(gap_hours, gap_weekdays):
            similar_data = energy_ts_processed[
                (energy_ts_processed['anomaly'] == 0) &  
                (energy_ts_processed['timestamp'].dt.hour == hour) & 
                (energy_ts_processed['timestamp'].dt.weekday == weekday)  
            ]
            
            if len(similar_data) > 0:
                avg_consumption = similar_data['meter_reading'].mean()
            else:
                avg_consumption = energy_ts_processed[energy_ts_processed['anomaly'] == 0]['meter_reading'].mean()
            
            estimated_values.append(avg_consumption)
        
        estimated_sum = sum(estimated_values)
        if imputed_sum > 0 and estimated_sum > 0:
            scaling_factor = estimated_sum / imputed_sum
        else:
            scaling_factor = 1.0
        
        print(f"  Lücke {gap_id}: Imputation={imputed_sum:.2f}, Schätzung={estimated_sum:.2f}, Skalierungsfaktor={scaling_factor:.4f}")
        scaling_factor = max(0.5, min(2.0, scaling_factor))
        imputed_ts.loc[
            (imputed_ts['timestamp'] >= gap_start) & 
            (imputed_ts['timestamp'] <= gap_end) &
            (imputed_ts['anomaly'] == 0),  
            'meter_reading'
        ]
        imputed_ts.loc[
            (imputed_ts['timestamp'] >= gap_start) & 
            (imputed_ts['timestamp'] <= gap_end) &
            (imputed_ts['anomaly'] == 0),  
            'meter_reading'
        ] *= scaling_factor
        imputed_ts.loc[imputed_ts['meter_reading'] < 0, 'meter_reading'] = 0
    
    print("Imputation abgeschlossen!")
    
    return imputed_ts

def visualize_imputation(original_df, imputed_df, window_size=168):
    anomaly_indices = original_df[original_df['anomaly'] == 1]['timestamp']
    
    if len(anomaly_indices) > 0:
        center_idx = anomaly_indices.iloc[len(anomaly_indices) // 2]  
        start_idx = center_idx - pd.Timedelta(hours=window_size//2)
        end_idx = center_idx + pd.Timedelta(hours=window_size//2)
        original_window = original_df[(original_df['timestamp'] >= start_idx) & (original_df['timestamp'] <= end_idx)]
        imputed_window = imputed_df[(imputed_df['timestamp'] >= start_idx) & (imputed_df['timestamp'] <= end_idx)]

        plt.figure(figsize=(15, 8))
        plt.plot(original_window['timestamp'], original_window['meter_reading'], 'b-', alpha=0.7, label='Original')
        plt.plot(imputed_window['timestamp'], imputed_window['meter_reading'], 'g-', label='Imputed')

        anomalies = original_window[original_window['anomaly'] == 1]
        plt.scatter(anomalies['timestamp'], anomalies['meter_reading'], c='r', s=50, label='Anomalies')
        imputed_points = imputed_window[original_window['anomaly'] == 1]
        plt.scatter(imputed_points['timestamp'], imputed_points['meter_reading'], c='g', s=80, marker='x', label='Imputed Points')
        
        plt.title('Original vs. Imputed Energy Consumption')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption (kWh)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        original_daily = original_window.groupby(original_window['timestamp'].dt.date)['meter_reading'].sum()
        imputed_daily = imputed_window.groupby(imputed_window['timestamp'].dt.date)['meter_reading'].sum()
        
        plt.figure(figsize=(15, 5))
        plt.bar(range(len(original_daily)), original_daily.values, alpha=0.6, label='Original')
        plt.bar(range(len(imputed_daily)), imputed_daily.values, alpha=0.6, label='Imputed')
        plt.xticks(range(len(original_daily)), original_daily.index, rotation=45)
        plt.title('Daily Energy Consumption: Original vs. Imputed')
        plt.xlabel('Date')
        plt.ylabel('Daily Energy (kWh)')
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
    else:
        print("Keine Anomalien zum Visualisieren gefunden!")

def evaluate_imputation(original_df, imputed_df):
    anomaly_mask = original_df['anomaly'] == 1
    if 'true_value' in original_df.columns:
        true_values = original_df.loc[anomaly_mask, 'true_value']
        imputed_values = imputed_df.loc[anomaly_mask, 'meter_reading']

        mae = np.mean(np.abs(imputed_values.values - true_values.values))
        mse = np.mean((imputed_values.values - true_values.values) ** 2)
        non_zero_mask = true_values != 0
        mape = np.mean(np.abs((true_values[non_zero_mask].values - imputed_values[non_zero_mask].values) / true_values[non_zero_mask].values)) * 100
        
        print(f"Bewertung der Imputation mit Ground Truth:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAPE: {mape:.2f}%")

    imputed_values = imputed_df.loc[anomaly_mask, 'meter_reading']
    
    print("\nStatistiken über die imputierten Werte:")
    print(f"Anzahl imputierter Werte: {len(imputed_values)}")
    print(f"Min: {imputed_values.min():.2f}")
    print(f"Max: {imputed_values.max():.2f}")
    print(f"Mittelwert: {imputed_values.mean():.2f}")
    print(f"Median: {imputed_values.median():.2f}")
    print(f"Standardabweichung: {imputed_values.std():.2f}")

    normal_values = original_df.loc[~anomaly_mask, 'meter_reading']
    
    print("\nStatistiken über die normalen (nicht-anomalen) Werte:")
    print(f"Anzahl normaler Werte: {len(normal_values)}")
    print(f"Min: {normal_values.min():.2f}")
    print(f"Max: {normal_values.max():.2f}")
    print(f"Mittelwert: {normal_values.mean():.2f}")
    print(f"Median: {normal_values.median():.2f}")
    print(f"Standardabweichung: {normal_values.std():.2f}")

    negative_values = imputed_values[imputed_values < 0]
    if len(negative_values) > 0:
        print(f"\nWARNUNG: {len(negative_values)} negative imputierte Werte gefunden!")
        print(f"Min negativer Wert: {negative_values.min():.2f}")
        print(f"Diese sollten in der finalen Version korrigiert werden.")
    else:
        print("\nKeine negativen imputierten Werte gefunden - sehr gut!")

def main(file_path):
    print("Lade Daten...")
    df = load_data(file_path)
    print(f"Insgesamt {len(df)} Datenpunkte geladen.")
    print(f"Gefundene Anomalien: {df['anomaly'].sum()}")
    print("Führe DD-CPI Imputation durch...")

    imputed_df = dd_cpi_imputation(df)
    print("Evaluiere die Imputation...")
    evaluate_imputation(df, imputed_df)
    print("Visualisiere die Ergebnisse...")
    visualize_imputation(df, imputed_df)
    print("Speichere imputierte Daten...")

    imputed_df.to_csv('id118neu/imputed_meter_readings_118_CPI_mehr_1.csv', index=False)
    print("Fertig! Imputierte Daten wurden in 'imputed_meter_readings.csv' gespeichert.")

if __name__ == "__main__":
    file_path = "id439/filtered_data_building_439_filled_mehr_1.csv" 
    main(file_path)