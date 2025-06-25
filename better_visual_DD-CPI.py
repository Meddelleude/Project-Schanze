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

def dd_cpi_imputation_with_tracking(energy_ts, weights=(5, 1, 10)):
    """
    Erweiterte Version der DD-CPI Imputation die Template-Matches verfolgt
    """
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
    
    # Dictionary zum Verfolgen der Template-Matches
    template_matches = {}
    
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
        
        # Template-Match speichern
        template_matches[day_with_gaps['date']] = best_matching_day['date']
        
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
        ] *= scaling_factor
        
        imputed_ts.loc[imputed_ts['meter_reading'] < 0, 'meter_reading'] = 0
    
    print("Imputation abgeschlossen!")
    
    return imputed_ts, template_matches

def visualize_imputation_with_template(original_df, imputed_df, template_matches=None, window_size=168):
    """
    Visualisiert die Imputation inklusive der verwendeten Kopiervorlagen
    
    Args:
        original_df: Original DataFrame
        imputed_df: Imputierter DataFrame  
        template_matches: Dictionary mit {gap_date: template_date} Mappings
        window_size: Größe des Visualisierungsfensters in Stunden
    """
    anomaly_indices = original_df[original_df['anomaly'] == 1]['timestamp']
    
    if len(anomaly_indices) > 0:
        # Finde die Mitte der Anomalien für die Visualisierung
        center_idx = anomaly_indices.iloc[len(anomaly_indices) // 2]
        start_idx = center_idx - pd.Timedelta(hours=window_size//2)
        end_idx = center_idx + pd.Timedelta(hours=window_size//2)
        
        original_window = original_df[(original_df['timestamp'] >= start_idx) & (original_df['timestamp'] <= end_idx)]
        imputed_window = imputed_df[(imputed_df['timestamp'] >= start_idx) & (imputed_df['timestamp'] <= end_idx)]

        # Plot 1: Hauptdiagramm - Original vs. Imputed
        fig1, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        
        ax1.plot(original_window['timestamp'], original_window['meter_reading'], 'b-', alpha=0.7, label='Original', linewidth=1.5)
        ax1.plot(imputed_window['timestamp'], imputed_window['meter_reading'], 'g-', label='Imputed', linewidth=1.5)

        anomalies = original_window[original_window['anomaly'] == 1]
        ax1.scatter(anomalies['timestamp'], anomalies['meter_reading'], c='r', s=50, label='Anomalies', zorder=5)
        
        imputed_points = imputed_window[original_window['anomaly'] == 1]
        ax1.scatter(imputed_points['timestamp'], imputed_points['meter_reading'], 
                   c='g', s=80, marker='x', label='Imputed Points', zorder=5)
        
        ax1.set_title('Original vs. Imputed Energy Consumption', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Energy Consumption (kWh)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speichere Plot 1
        timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename1 = f'dd_cpi_main_plot_{timestamp_str}.png'
        fig1.savefig(filename1, dpi=300, bbox_inches='tight')
        print(f"Hauptdiagramm gespeichert als: {filename1}")
        plt.show()
        
        # Berechne die Zeitspannenlänge UND Y-Achsenlimits vom ersten Diagramm
        x_min, x_max = ax1.get_xlim()
        y_min, y_max = ax1.get_ylim()
        time_span = x_max - x_min  # Länge der Zeitspanne in Tagen
        
        # Plot 2: Template-Tage anzeigen
        if template_matches:
            # Finde Template-Tage die in unserem Fenster verwendet wurden
            window_dates = set(original_window['timestamp'].dt.date)
            relevant_templates = {}
            
            for gap_date, template_date in template_matches.items():
                if gap_date.date() in window_dates:
                    relevant_templates[gap_date] = template_date
            
            if relevant_templates:
                # Jeder Template-Tag bekommt sein eigenes großes Fenster
                for i, (gap_date, template_date) in enumerate(relevant_templates.items()):
                    fig_template, ax_template = plt.subplots(1, 1, figsize=(15, 8))
                    
                    # Zeige den Template-Tag in seinem eigenen Zeitbereich
                    template_day_start = template_date.normalize()
                    template_day_end = template_day_start + timedelta(days=1)
                    template_data_full = original_df[
                        (original_df['timestamp'] >= template_day_start) & 
                        (original_df['timestamp'] < template_day_end) &
                        (original_df['anomaly'] == 0)
                    ]
                    
                    if len(template_data_full) > 0:
                        ax_template.plot(template_data_full['timestamp'], template_data_full['meter_reading'], 
                                       'b-o', linewidth=2, markersize=4,
                                       label=f'Template Day: {template_date.strftime("%Y-%m-%d (%A)")}')
                    
                    # Setze X-Achse auf die gleiche ZEITSPANNENLÄNGE, aber zentriert um den Template-Tag
                    template_center = template_day_start + timedelta(hours=12)  # Mitte des Template-Tages
                    half_span = timedelta(days=time_span/2)
                    
                    template_x_min = template_center - half_span
                    template_x_max = template_center + half_span
                    
                    ax_template.set_xlim(template_x_min, template_x_max)
                    
                    # Setze Y-Achse auf die gleiche Skalierung wie das erste Diagramm
                    ax_template.set_ylim(y_min, y_max)
                    
                    # Einfacher Titel: nur Gap-Info und origineller Wochentag
                    ax_template.set_title(f'Used for Gap: {gap_date.strftime("%Y-%m-%d (%A)")}', 
                                        fontsize=14, fontweight='bold')
                    ax_template.set_xlabel('Time')
                    ax_template.set_ylabel('Energy Consumption (kWh)')
                    ax_template.legend()
                    ax_template.grid(True, alpha=0.3)
                    ax_template.tick_params(axis='x', rotation=45)
                    
                    # Speichere Template-Plot
                    filename_template = f'dd_cpi_template_{i+1}_{template_date.strftime("%Y%m%d")}_{timestamp_str}.png'
                    fig_template.savefig(filename_template, dpi=300, bbox_inches='tight')
                    print(f"Template-Diagramm {i+1} gespeichert als: {filename_template}")
                    print(f"  Zeitspanne: {template_x_min.strftime('%Y-%m-%d %H:%M')} bis {template_x_max.strftime('%Y-%m-%d %H:%M')} (gleiche Länge: {time_span:.1f} Tage)")
                    print(f"  Y-Achsen-Range: {y_min:.2f} bis {y_max:.2f} (identisch mit Hauptdiagramm)")
                    plt.show()
            else:
                print("Keine Template-Matches im aktuellen Zeitfenster gefunden.")
        else:
            print("Template-Matching-Informationen nicht verfügbar.")
        
        # Zusätzliche Informationen ausgeben
        if template_matches:
            print("\n" + "="*60)
            print("TEMPLATE MATCHING INFORMATION")
            print("="*60)
            for gap_date, template_date in template_matches.items():
                print(f"Gap Date: {gap_date.date()} -> Template Date: {template_date.date()}")
                print(f"  Gap Weekday: {gap_date.strftime('%A')}")
                print(f"  Template Weekday: {template_date.strftime('%A')}")
                print(f"  Days apart: {abs((gap_date - template_date).days)} days")
                print("-" * 40)
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

def main_enhanced(file_path):
    """
    Erweiterte Hauptfunktion mit Template-Tracking
    """
    print("Lade Daten...")
    df = load_data(file_path)
    print(f"Insgesamt {len(df)} Datenpunkte geladen.")
    print(f"Gefundene Anomalien: {df['anomaly'].sum()}")
    
    print("Führe DD-CPI Imputation durch...")
    imputed_df, template_matches = dd_cpi_imputation_with_tracking(df)
    
    print("Evaluiere die Imputation...")
    evaluate_imputation(df, imputed_df)
    
    print("Visualisiere die Ergebnisse...")
    visualize_imputation_with_template(df, imputed_df, template_matches)
    
    print("Speichere imputierte Daten...")
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Speichere imputierte Daten mit Zeitstempel
    imputed_filename = f'imputed_meter_readings_{timestamp_str}.csv'
    imputed_df.to_csv(imputed_filename, index=False)
    print(f"Imputierte Daten gespeichert als: {imputed_filename}")
    
    # Speichere auch die Template-Matches
    if template_matches:
        template_df = pd.DataFrame([
            {'gap_date': gap_date, 'template_date': template_date, 
             'gap_weekday': gap_date.strftime('%A'), 'template_weekday': template_date.strftime('%A'),
             'days_apart': abs((gap_date - template_date).days)}
            for gap_date, template_date in template_matches.items()
        ])
        template_filename = f'template_matches_{timestamp_str}.csv'
        template_df.to_csv(template_filename, index=False)
        print(f"Template-Matches gespeichert als: {template_filename}")
    
    print(f"\nAlle Ausgabedateien wurden mit Zeitstempel {timestamp_str} erstellt:")
    print(f"  - {imputed_filename}")
    if template_matches:
        print(f"  - {template_filename}")
    print(f"  - dd_cpi_imputation_results_{timestamp_str}.png")
    
    print("Fertig!")

# Zusätzliche Hilfsfunktion für detaillierte Template-Analyse
def analyze_template_quality(original_df, template_matches):
    """
    Analysiert die Qualität der Template-Matches
    """
    if not template_matches:
        print("Keine Template-Matches zum Analysieren gefunden.")
        return
    
    print("\n" + "="*70)
    print("DETAILLIERTE TEMPLATE-QUALITÄTSANALYSE")
    print("="*70)
    
    weekday_matches = {'exact': 0, 'same_type': 0, 'different': 0}
    energy_differences = []
    seasonal_differences = []
    
    for gap_date, template_date in template_matches.items():
        # Wochentag-Analyse
        gap_weekday = gap_date.weekday()
        template_weekday = template_date.weekday()
        
        if gap_weekday == template_weekday:
            weekday_matches['exact'] += 1
        elif (gap_weekday < 5 and template_weekday < 5) or (gap_weekday >= 5 and template_weekday >= 5):
            weekday_matches['same_type'] += 1
        else:
            weekday_matches['different'] += 1
        
        # Saisonale Unterschiede
        gap_day_of_year = gap_date.timetuple().tm_yday
        template_day_of_year = template_date.timetuple().tm_yday
        seasonal_diff = min(abs(gap_day_of_year - template_day_of_year), 
                           365 - abs(gap_day_of_year - template_day_of_year))
        seasonal_differences.append(seasonal_diff)
        
        # Energie-Unterschiede (falls verfügbar)
        gap_day_start = gap_date.normalize()
        gap_day_end = gap_day_start + timedelta(days=1)
        template_day_start = template_date.normalize()
        template_day_end = template_day_start + timedelta(days=1)
        
        gap_energy = original_df[
            (original_df['timestamp'] >= gap_day_start) & 
            (original_df['timestamp'] < gap_day_end) &
            (original_df['anomaly'] == 0)
        ]['meter_reading'].sum()
        
        template_energy = original_df[
            (original_df['timestamp'] >= template_day_start) & 
            (original_df['timestamp'] < template_day_end) &
            (original_df['anomaly'] == 0)
        ]['meter_reading'].sum()
        
        if template_energy > 0:
            energy_diff_percent = abs(gap_energy - template_energy) / template_energy * 100
            energy_differences.append(energy_diff_percent)
    
    # Ergebnisse ausgeben
    total_matches = len(template_matches)
    print(f"Gesamte Template-Matches: {total_matches}")
    print(f"\nWochentag-Matching:")
    print(f"  Exakte Übereinstimmung: {weekday_matches['exact']} ({weekday_matches['exact']/total_matches*100:.1f}%)")
    print(f"  Gleicher Typ (Wochentag/Wochenende): {weekday_matches['same_type']} ({weekday_matches['same_type']/total_matches*100:.1f}%)")
    print(f"  Unterschiedlicher Typ: {weekday_matches['different']} ({weekday_matches['different']/total_matches*100:.1f}%)")
    
    if seasonal_differences:
        print(f"\nSaisonale Unterschiede:")
        print(f"  Durchschnitt: {np.mean(seasonal_differences):.1f} Tage")
        print(f"  Median: {np.median(seasonal_differences):.1f} Tage")
        print(f"  Min: {min(seasonal_differences)} Tage")
        print(f"  Max: {max(seasonal_differences)} Tage")
    
    if energy_differences:
        print(f"\nEnergie-Unterschiede:")
        print(f"  Durchschnittliche Abweichung: {np.mean(energy_differences):.1f}%")
        print(f"  Median Abweichung: {np.median(energy_differences):.1f}%")
        print(f"  Min Abweichung: {min(energy_differences):.1f}%")
        print(f"  Max Abweichung: {max(energy_differences):.1f}%")

def create_template_comparison_plot(original_df, template_matches, max_comparisons=5):
    """
    Erstellt detaillierte Vergleichsplots zwischen Gap-Tagen und ihren Templates
    """
    if not template_matches:
        print("Keine Template-Matches zum Plotten gefunden.")
        return
    
    # Begrenze die Anzahl der Vergleiche für bessere Lesbarkeit
    matches_to_plot = list(template_matches.items())[:max_comparisons]
    
    fig, axes = plt.subplots(len(matches_to_plot), 1, figsize=(15, 4*len(matches_to_plot)))
    if len(matches_to_plot) == 1:
        axes = [axes]
    
    for i, (gap_date, template_date) in enumerate(matches_to_plot):
        # Template-Tag Daten
        template_day_start = template_date.normalize()
        template_day_end = template_day_start + timedelta(days=1)
        template_data = original_df[
            (original_df['timestamp'] >= template_day_start) & 
            (original_df['timestamp'] < template_day_end) &
            (original_df['anomaly'] == 0)
        ]
        
        # Gap-Tag Daten (normale Werte falls vorhanden)
        gap_day_start = gap_date.normalize()
        gap_day_end = gap_day_start + timedelta(days=1)
        gap_data_normal = original_df[
            (original_df['timestamp'] >= gap_day_start) & 
            (original_df['timestamp'] < gap_day_end) &
            (original_df['anomaly'] == 0)
        ]
        
        # Plot Template-Tag
        if len(template_data) > 0:
            template_hours = template_data['timestamp'].dt.hour
            axes[i].plot(template_hours, template_data['meter_reading'], 
                        'b-o', linewidth=2, markersize=4, alpha=0.8,
                        label=f'Template: {template_date.strftime("%Y-%m-%d (%A)")}')
        
        # Plot Gap-Tag (normale Werte)
        if len(gap_data_normal) > 0:
            gap_hours = gap_data_normal['timestamp'].dt.hour
            axes[i].plot(gap_hours, gap_data_normal['meter_reading'], 
                        'g-s', linewidth=2, markersize=4, alpha=0.8,
                        label=f'Gap Day (normal): {gap_date.strftime("%Y-%m-%d (%A)")}')
        
        # Gap-Tag Anomalien
        gap_data_anomaly = original_df[
            (original_df['timestamp'] >= gap_day_start) & 
            (original_df['timestamp'] < gap_day_end) &
            (original_df['anomaly'] == 1)
        ]
        
        if len(gap_data_anomaly) > 0:
            anomaly_hours = gap_data_anomaly['timestamp'].dt.hour
            axes[i].scatter(anomaly_hours, gap_data_anomaly['meter_reading'], 
                           c='red', s=50, marker='x', alpha=0.8,
                           label=f'Missing values')
        
        axes[i].set_title(f'Comparison {i+1}: Gap vs Template Day')
        axes[i].set_xlabel('Hour of Day')
        axes[i].set_ylabel('Energy Consumption (kWh)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 23)
    
    plt.tight_layout()
    
    # Speichere die Vergleichsplots
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f'template_comparison_details_{timestamp_str}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Template-Vergleichsdiagramm gespeichert als: {filename}")
    
    plt.show()

if __name__ == "__main__":
    # Passen Sie den Dateipfad an Ihre Daten an
    file_path = "id685/mit_missing_markierung_building_685.csv" 
    
    # Führe die erweiterte Imputation durch
    main_enhanced(file_path)
    
    # Optional: Lade die Ergebnisse für weitere Analysen
    # df_original = load_data(file_path)
    # df_imputed = pd.read_csv('imputed_meter_readings_enhanced.csv')
    # template_matches_df = pd.read_csv('template_matches.csv')
    
    # Konvertiere Template-Matches zurück zum Dictionary-Format für weitere Analysen
    # template_matches = {}
    # for _, row in template_matches_df.iterrows():
    #     gap_date = pd.to_datetime(row['gap_date'])
    #     template_date = pd.to_datetime(row['template_date'])
    #     template_matches[gap_date] = template_date
    
    # Führe detaillierte Analysen durch
    # analyze_template_quality(df_original, template_matches)
    # create_template_comparison_plot(df_original, template_matches)