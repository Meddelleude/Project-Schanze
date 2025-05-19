import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')  # Prophet gibt viele Warnungen aus

def load_data(file_path):
    """
    Lädt die CSV-Datei mit Zeitstempeln, Gebäude-ID, stündlichen Verbrauchswerten und Anomaliemarkierungen
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def single_value_linear_interpolation(energy_ts):
    """
    Interpoliert einzelne fehlende Werte in einer Energiezeitreihe linear
    """
    # Setze timestamp als Index für einfachere Operationen
    energy_ts_indexed = energy_ts.set_index('timestamp')
    
    # Identifiziere einzelne fehlende Werte (nicht Teil einer längeren Lücke)
    mask = energy_ts_indexed['anomaly'] == 1
    single_missing = mask & ~(mask.shift(1) & mask | mask & mask.shift(-1))
    
    # Kopiere die Zeitreihe
    interpolated_ts = energy_ts_indexed.copy()
    
    # Identifiziere die Indizes der einzelnen fehlenden Werte
    single_missing_idx = interpolated_ts.index[single_missing]
    
    print(f"Einzelne fehlende Werte gefunden: {len(single_missing_idx)}")
    
    for idx in single_missing_idx:
        # Finde den nächsten und vorherigen nicht-fehlenden Wert
        prev_idx = interpolated_ts.index[interpolated_ts.index < idx][-1] if any(interpolated_ts.index < idx) else None
        next_idx = interpolated_ts.index[interpolated_ts.index > idx][0] if any(interpolated_ts.index > idx) else None
        
        if prev_idx is not None and next_idx is not None:
            prev_value = interpolated_ts.loc[prev_idx, 'meter_reading']
            next_value = interpolated_ts.loc[next_idx, 'meter_reading']
            
            # Lineare Interpolation für den fehlenden Wert
            time_diff_prev = (idx - prev_idx).total_seconds() / 3600  # Umrechnung in Stunden
            time_diff_total = (next_idx - prev_idx).total_seconds() / 3600
            
            interpolated_value = prev_value + (next_value - prev_value) * (time_diff_prev / time_diff_total)
            
            # Stelle sicher, dass der interpolierte Wert nicht negativ ist
            interpolated_value = max(0, interpolated_value)
            
            interpolated_ts.loc[idx, 'meter_reading'] = interpolated_value
            interpolated_ts.loc[idx, 'anomaly'] = 0  # Markiere als nicht mehr anomal
    
    # Zurück zum originalen Format mit timestamp als Spalte
    return interpolated_ts.reset_index()

def calculate_energy_per_day(energy_ts):
    """
    Berechnet die tägliche Energiesumme aus den stündlichen Verbrauchswerten
    """
    # Konvertiere timestamp zu Datum
    energy_ts['date'] = energy_ts['timestamp'].dt.date
    
    # Gruppiere nach Tagen
    daily_energy = energy_ts.groupby('date').agg({
        'meter_reading': 'sum',  # Summe der stündlichen Werte für den Tag
        'anomaly': lambda x: 1 if any(x == 1) else 0,  # Tag hat Anomalien, wenn mindestens ein Wert anomal ist
        'building_id': 'first'  # Gebäude-ID beibehalten
    }).reset_index()
    
    # Konvertiere 'date' zurück zu timestamp und setze als Index
    daily_energy['timestamp'] = pd.to_datetime(daily_energy['date'])
    daily_energy.drop('date', axis=1, inplace=True)
    daily_energy.set_index('timestamp', inplace=True)
    
    print(f"Tägliche Energiewerte berechnet: {len(daily_energy)} Tage")
    print(f"Davon Tage mit Anomalien: {daily_energy['anomaly'].sum()}")
    
    return daily_energy

def determine_days_with_missing_values(energy_ts):
    """
    Ermittelt, welche Tage fehlende Werte haben
    """
    # Konvertiere timestamp zu Datum
    energy_ts['date'] = energy_ts['timestamp'].dt.date
    
    # Gruppiere nach Tagen und prüfe, ob es anomale Werte gibt
    days_with_missing = energy_ts.groupby('date')['anomaly'].max()
    
    print(f"Tage mit fehlenden Werten ermittelt: {days_with_missing.sum()} von {len(days_with_missing)}")
    
    return days_with_missing.values

def estimate_weekly_pattern_with_prophet(daily_energy, non_complete_days):
    """
    Schätzt das wöchentliche Muster im Energieverbrauch mit Prophet
    """
    print(f"Estimating weekly pattern with Prophet...")
    print(f"Daily energy shape: {daily_energy.shape}")
    print(f"Non-complete days shape: {non_complete_days.shape}")
    
    # Bereite Daten für Prophet vor - nur vollständige Tage verwenden
    complete_days = ~non_complete_days.astype(bool)
    print(f"Complete days: {sum(complete_days)} von {len(complete_days)}")
    
    # Prüfe, ob es überhaupt vollständige Tage gibt
    if not any(complete_days):
        print("WARNUNG: Keine vollständigen Tage zum Trainieren des Prophet-Modells gefunden.")
        print("Verwende einen einfachen Wochentag-basierten Fallback-Ansatz.")
        
        # Erzeuge ein einfaches Wochenmuster als Fallback
        weekdays = [daily_energy.index[i].weekday() for i in range(len(daily_energy))]
        
        # Einfaches Wochenmuster erstellen (positiv für Wochentage, negativ für Wochenende)
        fallback_pattern = []
        for wd in weekdays:
            if wd < 5:  # Wochentag (Montag-Freitag)
                fallback_pattern.append(0.1)  # Leicht erhöhter Verbrauch an Wochentagen
            else:  # Wochenende (Samstag-Sonntag)
                fallback_pattern.append(-0.15)  # Leicht verringerter Verbrauch am Wochenende
        
        # Normalisieren, damit die Summe Null ergibt
        fallback_pattern = np.array(fallback_pattern)
        fallback_pattern = fallback_pattern - np.mean(fallback_pattern)
        
        return fallback_pattern
    
    # Erstelle ein DataFrame für Prophet
    prophet_df = daily_energy.reset_index().copy()
    
    # Filtere Tage ohne Anomalien
    prophet_df = prophet_df[complete_days]
    
    print(f"Prophet DataFrame shape nach Filterung: {prophet_df.shape}")
    
    # Benenne Spalten um, wie von Prophet erwartet
    prophet_df = prophet_df.rename(columns={'timestamp': 'ds', 'meter_reading': 'y'})
    
    # Debugging: Zeige die ersten Zeilen
    print("Erste Zeilen des Prophet DataFrame:")
    print(prophet_df.head())
    
    # Prüfe erneut, ob nach der Filterung Daten übrig bleiben
    if len(prophet_df) == 0:
        print("WARNUNG: Nach Filterung keine Daten für Prophet übrig.")
        # Verwende den gleichen Fallback wie oben
        weekdays = [daily_energy.index[i].weekday() for i in range(len(daily_energy))]
        
        fallback_pattern = []
        for wd in weekdays:
            if wd < 5:  # Wochentag (Montag-Freitag)
                fallback_pattern.append(0.1)  # Leicht erhöhter Verbrauch an Wochentagen
            else:  # Wochenende (Samstag-Sonntag)
                fallback_pattern.append(-0.15)  # Leicht verringerter Verbrauch am Wochenende
        
        fallback_pattern = np.array(fallback_pattern)
        fallback_pattern = fallback_pattern - np.mean(fallback_pattern)
        
        return fallback_pattern
    
    try:
        # Prüfe, ob die Daten für Prophet korrekt sind
        print("Prophet Daten überprüfen:")
        print(f"Sind NaN-Werte vorhanden? {prophet_df['y'].isna().any()}")
        
        # Entferne NaN-Werte
        prophet_df = prophet_df.dropna(subset=['y'])
        
        print(f"Prophet DataFrame shape nach NaN-Entfernung: {prophet_df.shape}")
        
        if len(prophet_df) == 0:
            raise ValueError("Alle Werte sind NaN!")
        
        # Trainiere Prophet-Modell
        model = Prophet(weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df[['ds', 'y']])
        
        # Erstelle Vorhersagen für alle Tage
        future = pd.DataFrame({'ds': daily_energy.index.tolist()})
        
        print(f"Future DataFrame shape: {future.shape}")
        print("Erste Zeilen des Future DataFrame:")
        print(future.head())
        
        forecast = model.predict(future)
        
        # Extrahiere das wöchentliche Muster
        weekly_pattern = forecast['weekly']
        
        print(f"Weekly pattern berechnet mit Länge: {len(weekly_pattern)}")
        
        return weekly_pattern.values
    except Exception as e:
        print(f"Fehler beim Trainieren des Prophet-Modells: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        print("Verwende einen Fallback-Ansatz.")
        
        # Fallback: Einfaches Wochenmuster basierend auf Wochentag
        weekdays = [daily_energy.index[i].weekday() for i in range(len(daily_energy))]
        
        fallback_pattern = []
        for wd in weekdays:
            if wd < 5:  # Wochentag (Montag-Freitag)
                fallback_pattern.append(0.1)  # Leicht erhöhter Verbrauch an Wochentagen
            else:  # Wochenende (Samstag-Sonntag)
                fallback_pattern.append(-0.15)  # Leicht verringerter Verbrauch am Wochenende
        
        fallback_pattern = np.array(fallback_pattern)
        fallback_pattern = fallback_pattern - np.mean(fallback_pattern)
        
        return fallback_pattern

def estimate_missing_energy_per_day(energy_ts, weekly_pattern):
    """
    Schätzt die fehlende Energie für Tage mit Lücken basierend auf stündlichen Verbrauchswerten
    """
    # Erstelle eine Serie mit Nullen für jeden Tag im Datensatz
    days = pd.date_range(
        start=energy_ts['timestamp'].min().normalize(), 
        end=energy_ts['timestamp'].max().normalize(),
        freq='D'
    )
    missing_energy = pd.Series(0, index=days)
    
    # Identifiziere Lücken (zusammenhängende anomale Werte)
    energy_ts['gap_id'] = (energy_ts['anomaly'].diff() == 1).cumsum()
    gaps = energy_ts[energy_ts['anomaly'] == 1].groupby('gap_id')
    
    print(f"Anzahl identifizierter Lücken: {len(gaps)}")
    
    for gap_id, gap_data in gaps:
        # Berechne Beginn und Ende der Lücke
        gap_start = gap_data['timestamp'].min()
        gap_end = gap_data['timestamp'].max()
        
        print(f"Verarbeite Lücke {gap_id}: {gap_start} bis {gap_end}")
        
        # Berechne die geschätzten stündlichen Verbrauchswerte für die Lücke
        # basierend auf ähnlichen Stunden an ähnlichen Tagen
        
        # Extrahiere Wochentag und Stunde aus den Lückendaten
        gap_hours = gap_data['timestamp'].dt.hour.values
        gap_weekdays = gap_data['timestamp'].dt.weekday.values
        
        # Finde ähnliche Stunden an ähnlichen Tagen (gleiche Stunde, gleicher Wochentag)
        similar_hours_data = []
        for hour, weekday in zip(gap_hours, gap_weekdays):
            similar_data = energy_ts[
                (energy_ts['anomaly'] == 0) &  # Nur normale Daten
                (energy_ts['timestamp'].dt.hour == hour) &  # Gleiche Stunde
                (energy_ts['timestamp'].dt.weekday == weekday)  # Gleicher Wochentag
            ]
            
            if len(similar_data) > 0:
                avg_consumption = similar_data['meter_reading'].mean()
            else:
                # Fallback: Gleiche Stunde, beliebiger Wochentag
                similar_time = energy_ts[
                    (energy_ts['anomaly'] == 0) &
                    (energy_ts['timestamp'].dt.hour == hour)
                ]
                
                if len(similar_time) > 0:
                    avg_consumption = similar_time['meter_reading'].mean()
                else:
                    # Letzter Fallback: Globaler Durchschnitt
                    avg_consumption = energy_ts[energy_ts['anomaly'] == 0]['meter_reading'].mean()
            
            similar_hours_data.append(avg_consumption)
        
        # Berechne die geschätzte Gesamtenergie für die Lücke
        estimated_gap_energy = sum(similar_hours_data)
        
        print(f"  Geschätzte Energie während der Lücke: {estimated_gap_energy:.2f}")
        
        # Tägliche Verteilung dieser Energie
        gap_days = pd.date_range(gap_start.normalize(), gap_end.normalize(), freq='D')
        days_in_gap = len(gap_days)
        
        print(f"  Tage in der Lücke: {days_in_gap}")
        
        # Berechne, wie viele fehlende Werte in jedem Tag der Lücke liegen
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
        
        # Verteile die geschätzte Energie basierend auf der Anzahl fehlender Werte pro Tag
        for day, count in missing_per_day.items():
            # Gewichteter Anteil der geschätzten Energie
            day_energy_share = estimated_gap_energy * (count / total_missing) if total_missing > 0 else 0
            
            # Füge den Energiewert zum entsprechenden Tag hinzu
            if day in missing_energy.index:
                missing_energy.loc[day] += day_energy_share
        
        # Passe die Verteilung basierend auf dem wöchentlichen Muster an (für mehrtägige Lücken)
        if days_in_gap > 1 and len(weekly_pattern) > 0:
            # Extrahiere das wöchentliche Muster für die Tage in der Lücke
            day_indices = [i % 7 for i in range(len(gap_days))]  # 0-6 für Montag-Sonntag
            
            # Stelle sicher, dass weekly_pattern lang genug ist
            if len(weekly_pattern) >= 7:
                weekly_adjustments = np.array([weekly_pattern[idx] for idx in day_indices])
            else:
                # Fallback, wenn weekly_pattern zu kurz ist
                weekly_adjustments = np.zeros(len(gap_days))
                print("  WARNUNG: weekly_pattern zu kurz, verwende Nullen als Anpassung")
            
            # Normalisiere die Anpassungen, damit die Summe Null ergibt
            if len(weekly_adjustments) > 0:
                weekly_adjustments = weekly_adjustments - np.mean(weekly_adjustments)
            
            # Wende die Anpassungen auf die Energieverteilung an
            for i, day in enumerate(gap_days):
                if day in missing_energy.index and i < len(weekly_adjustments):
                    missing_energy.loc[day] += weekly_adjustments[i]
    
    return missing_energy

def compile_list_of_complete_days(timestamps, daily_energy, non_complete_days):
    """
    Erstellt eine Liste von vollständigen Tagen mit ihren Merkmalen: 
    Gesamtenergie, Wochentag und saisonale Position
    """
    complete_days = []
    
    for i, (timestamp, is_incomplete) in enumerate(zip(timestamps, non_complete_days)):
        if not is_incomplete and i < len(daily_energy):
            # Extrahiere Merkmale des Tages
            energy = daily_energy.iloc[i]['meter_reading']
            
            if not pd.isna(energy):
                weekday = timestamp.weekday() + 1  # 1-7, wobei 1 = Montag
                # Saisonale Position (1-366) basierend auf Tag des Jahres
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
    """
    Findet den Tag mit der minimalen Unähnlichkeit zum Tag mit Lücken
    """
    we, ww, ws = weights  # Gewichte für Energie, Wochentag und Saison
    
    gap_day_date = day_with_gaps['date']
    gap_day_energy = day_with_gaps['energy']
    gap_day_weekday = gap_day_date.weekday() + 1  # 1-7
    gap_day_seasonal = gap_day_date.timetuple().tm_yday  # 1-366
    
    min_dissimilarity = float('inf')
    best_matching_day = None
    
    # Finde Min und Max der Energiewerte für Normalisierung
    energy_values = [day['energy'] for day in complete_days]
    e_min, e_max = min(energy_values), max(energy_values)
    e_range = e_max - e_min if e_max > e_min else 1  # Vermeide Division durch Null
    
    for complete_day in complete_days:
        # Berechne die drei Distanzmaße
        # 1. Energiedistanz
        de = abs(gap_day_energy - complete_day['energy']) / e_range
        
        # 2. Wochentagsdistanz
        if gap_day_weekday == complete_day['weekday']:
            dw = 0.0
        elif (gap_day_weekday <= 5 and complete_day['weekday'] <= 5) or \
             (gap_day_weekday >= 6 and complete_day['weekday'] >= 6):
            dw = 0.5  # Gleiche Kategorie (Arbeitstag oder Wochenende)
        else:
            dw = 1.0  # Unterschiedliche Kategorien
        
        # 3. Saisonale Distanz
        s = 366  # Für Schaltjahre
        seasonal_diff = abs(gap_day_seasonal - complete_day['seasonal_pos'])
        
        if seasonal_diff <= s / 2:
            ds = seasonal_diff / (s / 2)
        else:
            ds = (s - seasonal_diff) / (s / 2)
        
        # Kombiniere die Distanzmaße mit den Gewichten
        dissimilarity = we * de + ww * dw + ws * ds
        
        if dissimilarity < min_dissimilarity:
            min_dissimilarity = dissimilarity
            best_matching_day = complete_day
    
    return best_matching_day

def dd_cpi_imputation(energy_ts, weights=(5, 1, 10)):
    """
    Implementierung der Data-Driven Copy-Paste Imputation, angepasst für stündliche Verbrauchswerte
    
    Parameter:
    - energy_ts: DataFrame mit timestamp, meter_reading und anomaly Spalten
    - weights: Gewichte für die Unähnlichkeitskriterien (Energie, Wochentag, Saison)
    
    Returns:
    - DataFrame mit imputierten Werten
    """
    # Schritt 1: Lineare Interpolation einzelner fehlender Werte
    print("Schritt 1: Lineare Interpolation einzelner fehlender Werte...")
    energy_ts_processed = single_value_linear_interpolation(energy_ts)
    
    # Schritt 2: Energieverbrauchsschätzung
    print("Schritt 2: Energieverbrauchsschätzung...")
    daily_energy = calculate_energy_per_day(energy_ts_processed)
    non_complete_days = determine_days_with_missing_values(energy_ts_processed)
    weekly_pattern = estimate_weekly_pattern_with_prophet(daily_energy, non_complete_days)
    missing_energy = estimate_missing_energy_per_day(energy_ts_processed, weekly_pattern)
    
    # Kombiniere die geschätzte mit der tatsächlichen Energie
    estimated_energy_per_day = daily_energy.copy()
    for day in missing_energy.index:
        if day in estimated_energy_per_day.index:
            estimated_energy_per_day.loc[day, 'meter_reading'] += missing_energy.loc[day]
    
    # Schritt 3: Zusammenstellung der verfügbaren vollständigen Tage
    print("Schritt 3: Zusammenstellung der verfügbaren vollständigen Tage...")
    complete_days = compile_list_of_complete_days(
        daily_energy.index, 
        daily_energy, 
        non_complete_days
    )
    
    # Schritt 4: Für die Imputation brauchen wir keine Umrechnung von Energy zu Power,
    # da wir bereits stündliche Verbrauchswerte haben
    print("Schritt 4: Vorbereitung für die Imputation...")
    # Erstelle eine Kopie der originalen Zeitreihe für die Imputation
    imputed_ts = energy_ts_processed.copy()
    
    # Schritt 5: Identifiziere Tage mit Lücken und finde beste Übereinstimmungen
    print("Schritt 5: Finde und kopiere die besten passenden Tage...")
    days_with_gaps = []
    for day_idx, is_incomplete in enumerate(non_complete_days):
        if is_incomplete and day_idx < len(daily_energy.index):
            day_date = daily_energy.index[day_idx]
            days_with_gaps.append({
                'date': day_date,
                'energy': estimated_energy_per_day.loc[day_date, 'meter_reading'] if day_date in estimated_energy_per_day.index else 0
            })
    
    # Gehe jeden Tag mit Lücken durch
    for day_with_gaps in days_with_gaps:
        best_matching_day = find_day_with_min_dissimilarity(day_with_gaps, complete_days, weights)
        
        if best_matching_day is None:
            print(f"WARNUNG: Kein passender Tag gefunden für {day_with_gaps['date']}")
            continue
        
        print(f"Bester passender Tag für {day_with_gaps['date']}: {best_matching_day['date']}")
        
        # Hole die stündlichen Werte des besten passenden Tages
        match_day_start = best_matching_day['date']
        match_day_end = match_day_start + timedelta(days=1)
        
        matching_day_data = energy_ts_processed[
            (energy_ts_processed['timestamp'] >= match_day_start) & 
            (energy_ts_processed['timestamp'] < match_day_end) &
            (energy_ts_processed['anomaly'] == 0)
        ]
        
        # Erstelle ein Mapping von Stunde zu Verbrauchswert
        hour_to_consumption = {}
        for _, row in matching_day_data.iterrows():
            hour_key = row['timestamp'].hour
            hour_to_consumption[hour_key] = row['meter_reading']
        
        # Ersetze die anomalen Werte im Tag mit Lücken
        gap_day_start = day_with_gaps['date']
        gap_day_end = gap_day_start + timedelta(days=1)
        
        gap_day_anomalies = imputed_ts[
            (imputed_ts['timestamp'] >= gap_day_start) & 
            (imputed_ts['timestamp'] < gap_day_end) &
            (imputed_ts['anomaly'] == 1)
        ]
        
        for _, row in gap_day_anomalies.iterrows():
            ts_hour = row['timestamp'].hour
            
            # Wenn wir einen passenden Wert haben, ersetze den anomalen Wert
            if ts_hour in hour_to_consumption:
                # Verwende den kopierten Wert vom besten passenden Tag
                imputed_value = hour_to_consumption[ts_hour]
                imputed_ts.loc[imputed_ts['timestamp'] == row['timestamp'], 'meter_reading'] = imputed_value
                imputed_ts.loc[imputed_ts['timestamp'] == row['timestamp'], 'anomaly'] = 0  # Entferne Anomalie-Flag
    
    # Schritt 6: Skaliere die imputierten Werte, um die geschätzte Energiesumme einzuhalten
    print("Schritt 6: Skaliere die imputierten Werte...")
    
    # Identifiziere Lücken (zusammenhängende anomale Werte im Original)
    energy_ts['gap_id'] = (energy_ts['anomaly'].diff() == 1).cumsum()
    gaps = energy_ts[energy_ts['anomaly'] == 1].groupby('gap_id')
    
    for gap_id, gap_data in gaps:
        # Berechne Beginn und Ende der Lücke
        gap_start = gap_data['timestamp'].min()
        gap_end = gap_data['timestamp'].max()
        
        # Extrahiere imputierte Werte für diese Lücke
        imputed_values = imputed_ts[
            (imputed_ts['timestamp'] >= gap_start) & 
            (imputed_ts['timestamp'] <= gap_end) &
            (imputed_ts['anomaly'] == 0)  # Nur bereits imputierte Werte
        ]['meter_reading']
        
        if len(imputed_values) == 0:
            print(f"  WARNUNG: Keine imputierten Werte gefunden für Lücke {gap_id}")
            continue
        
        # Berechne die Summe der imputierten Werte
        imputed_sum = imputed_values.sum()
        
        # Berechne die geschätzte Summe für diese Lücke basierend auf ähnlichen Stunden
        gap_hours = gap_data['timestamp'].dt.hour.values
        gap_weekdays = gap_data['timestamp'].dt.weekday.values
        
        estimated_values = []
        for hour, weekday in zip(gap_hours, gap_weekdays):
            similar_data = energy_ts_processed[
                (energy_ts_processed['anomaly'] == 0) &  # Nur normale Daten
                (energy_ts_processed['timestamp'].dt.hour == hour) &  # Gleiche Stunde
                (energy_ts_processed['timestamp'].dt.weekday == weekday)  # Gleicher Wochentag
            ]
            
            if len(similar_data) > 0:
                avg_consumption = similar_data['meter_reading'].mean()
            else:
                # Fallback: Globaler Durchschnitt
                avg_consumption = energy_ts_processed[energy_ts_processed['anomaly'] == 0]['meter_reading'].mean()
            
            estimated_values.append(avg_consumption)
        
        estimated_sum = sum(estimated_values)
        
        # Berechne Skalierungsfaktor
        if imputed_sum > 0 and estimated_sum > 0:
            scaling_factor = estimated_sum / imputed_sum
        else:
            scaling_factor = 1.0
        
        print(f"  Lücke {gap_id}: Imputation={imputed_sum:.2f}, Schätzung={estimated_sum:.2f}, Skalierungsfaktor={scaling_factor:.4f}")
        
        # Begrenze den Skalierungsfaktor auf sinnvolle Werte (z.B. 0.5 bis 2.0)
        scaling_factor = max(0.5, min(2.0, scaling_factor))
        
        # Skaliere die imputierten Werte
        imputed_ts.loc[
            (imputed_ts['timestamp'] >= gap_start) & 
            (imputed_ts['timestamp'] <= gap_end) &
            (imputed_ts['anomaly'] == 0),  # Nur bereits imputierte Werte
            'meter_reading'
        ]
# Skaliere die imputierten Werte
        imputed_ts.loc[
            (imputed_ts['timestamp'] >= gap_start) & 
            (imputed_ts['timestamp'] <= gap_end) &
            (imputed_ts['anomaly'] == 0),  # Nur bereits imputierte Werte
            'meter_reading'
        ] *= scaling_factor
        
        # Stelle sicher, dass keine negativen Werte entstehen
        imputed_ts.loc[imputed_ts['meter_reading'] < 0, 'meter_reading'] = 0
    
    print("Imputation abgeschlossen!")
    
    return imputed_ts

def visualize_imputation(original_df, imputed_df, window_size=168):
    """
    Visualisiert die originalen und imputierten Daten in einem bestimmten Zeitfenster
    """
    # Finde einen Zeitraum mit Anomalien
    anomaly_indices = original_df[original_df['anomaly'] == 1]['timestamp']
    
    if len(anomaly_indices) > 0:
        center_idx = anomaly_indices.iloc[len(anomaly_indices) // 2]  # Mittlere Anomalie für bessere Visualisierung
        start_idx = center_idx - pd.Timedelta(hours=window_size//2)
        end_idx = center_idx + pd.Timedelta(hours=window_size//2)
        
        # Filtere die Daten für das Zeitfenster
        original_window = original_df[(original_df['timestamp'] >= start_idx) & (original_df['timestamp'] <= end_idx)]
        imputed_window = imputed_df[(imputed_df['timestamp'] >= start_idx) & (imputed_df['timestamp'] <= end_idx)]
        
        # Erstelle die Visualisierung
        plt.figure(figsize=(15, 8))
        
        # Zeichne originale und imputierte Werte
        plt.plot(original_window['timestamp'], original_window['meter_reading'], 'b-', alpha=0.7, label='Original')
        plt.plot(imputed_window['timestamp'], imputed_window['meter_reading'], 'g-', label='Imputed')
        
        # Markiere Anomalien im Original
        anomalies = original_window[original_window['anomaly'] == 1]
        plt.scatter(anomalies['timestamp'], anomalies['meter_reading'], c='r', s=50, label='Anomalies')
        
        # Markiere die imputierten Werte
        imputed_points = imputed_window[original_window['anomaly'] == 1]
        plt.scatter(imputed_points['timestamp'], imputed_points['meter_reading'], c='g', s=80, marker='x', label='Imputed Points')
        
        plt.title('Original vs. Imputed Energy Consumption')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption (kWh)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Zeige die Grafik
        plt.show()
        
        # Zeige auch die tägliche Verbrauchssumme vor und nach der Imputation
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
    """
    Bewertet die Qualität der Imputation
    """
    # Isoliere die imputierten Werte (Stellen, die im Original Anomalien waren)
    anomaly_mask = original_df['anomaly'] == 1
    
    # Wenn wir Ground Truth für die Anomalien haben (für synthetische Tests)
    if 'true_value' in original_df.columns:
        # Berechne Fehlermetriken zwischen imputierten Werten und wahren Werten
        true_values = original_df.loc[anomaly_mask, 'true_value']
        imputed_values = imputed_df.loc[anomaly_mask, 'meter_reading']
        
        # Mittlerer absoluter Fehler (MAE)
        mae = np.mean(np.abs(imputed_values.values - true_values.values))
        
        # Mittlerer quadratischer Fehler (MSE)
        mse = np.mean((imputed_values.values - true_values.values) ** 2)
        
        # Mittlerer absoluter prozentualer Fehler (MAPE)
        non_zero_mask = true_values != 0
        mape = np.mean(np.abs((true_values[non_zero_mask].values - imputed_values[non_zero_mask].values) / true_values[non_zero_mask].values)) * 100
        
        print(f"Bewertung der Imputation mit Ground Truth:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAPE: {mape:.2f}%")
    
    # Berechne Statistiken über die imputierten Werte
    imputed_values = imputed_df.loc[anomaly_mask, 'meter_reading']
    
    print("\nStatistiken über die imputierten Werte:")
    print(f"Anzahl imputierter Werte: {len(imputed_values)}")
    print(f"Min: {imputed_values.min():.2f}")
    print(f"Max: {imputed_values.max():.2f}")
    print(f"Mittelwert: {imputed_values.mean():.2f}")
    print(f"Median: {imputed_values.median():.2f}")
    print(f"Standardabweichung: {imputed_values.std():.2f}")
    
    # Vergleiche mit den Statistiken der normalen (nicht-anomalen) Werte
    normal_values = original_df.loc[~anomaly_mask, 'meter_reading']
    
    print("\nStatistiken über die normalen (nicht-anomalen) Werte:")
    print(f"Anzahl normaler Werte: {len(normal_values)}")
    print(f"Min: {normal_values.min():.2f}")
    print(f"Max: {normal_values.max():.2f}")
    print(f"Mittelwert: {normal_values.mean():.2f}")
    print(f"Median: {normal_values.median():.2f}")
    print(f"Standardabweichung: {normal_values.std():.2f}")
    
    # Prüfe auf negative imputierte Werte
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
    imputed_df.to_csv('id685/imputed_meter_readings_685_CPI_mehr_1.csv', index=False)
    
    print("Fertig! Imputierte Daten wurden in 'imputed_meter_readings.csv' gespeichert.")

# Beispielaufruf
if __name__ == "__main__":
    file_path = "id685/lead1.0-small_building_685_filled.csv"  
    main(file_path)