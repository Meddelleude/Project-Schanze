import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')  # Prophet gibt viele Warnungen aus


def load_data(file_path):
    """
    Lädt die CSV-Datei mit Zeitstempeln, Gebäude-ID, Verbrauchswerten und Anomaliemarkierungen
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def single_value_linear_interpolation(energy_ts):
    """
    Interpoliert einzelne fehlende Werte in einer Energiezeitreihe linear
    """
    # Identifiziere einzelne fehlende Werte (nicht Teil einer längeren Lücke)
    mask = energy_ts['anomaly'] == 1
    single_missing = mask & ~(mask.shift(1) & mask | mask & mask.shift(-1))
    
    # Kopiere die Zeitreihe
    interpolated_ts = energy_ts.copy()
    
    # Identifiziere die Indizes der einzelnen fehlenden Werte
    single_missing_idx = energy_ts.index[single_missing]
    
    for idx in single_missing_idx:
        # Finde den nächsten und vorherigen nicht-fehlenden Wert
        prev_idx = energy_ts.index[energy_ts.index < idx][-1] if any(energy_ts.index < idx) else None
        next_idx = energy_ts.index[energy_ts.index > idx][0] if any(energy_ts.index > idx) else None
        
        if prev_idx is not None and next_idx is not None:
            prev_value = energy_ts.loc[prev_idx, 'meter_reading']
            next_value = energy_ts.loc[next_idx, 'meter_reading']
            
            # Lineare Interpolation für den fehlenden Wert
            time_diff_prev = (idx - prev_idx).total_seconds() / 3600  # Umrechnung in Stunden
            time_diff_total = (next_idx - prev_idx).total_seconds() / 3600
            
            interpolated_value = prev_value + (next_value - prev_value) * (time_diff_prev / time_diff_total)
            interpolated_ts.loc[idx, 'meter_reading'] = interpolated_value
            interpolated_ts.loc[idx, 'anomaly'] = 0  # Markiere als nicht mehr anomal
    
    return interpolated_ts

def calculate_energy_per_day(energy_ts):
    """
    Berechnet den täglichen Energieverbrauch aus der Energiezeitreihe
    """
    # Sicherstellen, dass timestamp als Datum interpretiert wird
    energy_ts['date'] = energy_ts['timestamp'].dt.date
    
    # Gruppiere nach Tagen
    daily_groups = energy_ts.groupby('date')
    
    # Leerer DataFrame für Ergebnisse
    daily_energy = pd.DataFrame(columns=['meter_reading', 'anomaly', 'building_id'])
    
    for day, group in daily_groups:
        if len(group) >= 2:  # Mindestens 2 Messungen pro Tag nötig
            # Berechne Energiedifferenz zwischen Anfangswert und Endwert des Tages
            start_value = group['meter_reading'].iloc[0]
            end_value = group['meter_reading'].iloc[-1]
            energy_diff = end_value - start_value
            
            # Überprüfe, ob der Tag Anomalien enthält
            has_anomaly = 1 if any(group['anomaly'] == 1) else 0
            
            # Speichere Ergebnis
            timestamp = pd.Timestamp(day)
            building_id = group['building_id'].iloc[0]
            
            new_row = pd.DataFrame({
                'timestamp': [timestamp],
                'meter_reading': [energy_diff],
                'anomaly': [has_anomaly],
                'building_id': [building_id]
            })
            
            daily_energy = pd.concat([daily_energy, new_row], ignore_index=True)
    
    # Setze timestamp als Index
    if not daily_energy.empty:
        daily_energy.set_index('timestamp', inplace=True)
    
    print(f"Tägliche Energiewerte berechnet: {len(daily_energy)} Tage")
    if len(daily_energy) == 0:
        print("WARNUNG: Keine täglichen Energiewerte berechnet!")
    else:
        print(f"Davon Tage mit Anomalien: {daily_energy['anomaly'].sum()}")
    
    return daily_energy

def determine_days_with_missing_values(energy_ts):
    """
    Ermittelt, welche Tage fehlende Werte haben
    """
    # Sicherstellen, dass timestamp als Datum interpretiert wird
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
        # Wochentage (0-6 für Montag bis Sonntag)
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
        
        # Erstelle Vorhersagen für alle Tage - KORRIGIERT
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
    Schätzt die fehlende Energie für Tage mit Lücken
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
        
        # Finde den Energiewert vor und nach der Lücke
        before_gap = energy_ts[energy_ts['timestamp'] < gap_start]
        after_gap = energy_ts[energy_ts['timestamp'] > gap_end]
        
        if len(before_gap) == 0 or len(after_gap) == 0:
            print(f"Lücke {gap_id} am Rand der Zeitreihe, überspringe...")
            continue
        
        before_gap_idx = before_gap['timestamp'].max()
        after_gap_idx = after_gap['timestamp'].min()
        
        before_gap_value = energy_ts[energy_ts['timestamp'] == before_gap_idx]['meter_reading'].values[0]
        after_gap_value = energy_ts[energy_ts['timestamp'] == after_gap_idx]['meter_reading'].values[0]
        
        # Berechne die tatsächliche Energiedifferenz während der Lücke
        total_gap_energy = after_gap_value - before_gap_value
        
        print(f"  Energiedifferenz während der Lücke: {total_gap_energy:.2f}")
        
        # Tägliche Verteilung dieser Energie basierend auf der Anzahl der fehlenden Werte pro Tag
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
        
        # Verteile die Energie nach Anteil der fehlenden Werte
        for day, count in missing_per_day.items():
            # Grundverteilung nach Anteil der fehlenden Werte
            day_energy = total_gap_energy * (count / total_missing) if total_missing > 0 else 0
            
            # Füge den Energiewert zum entsprechenden Tag hinzu
            if day in missing_energy.index:
                missing_energy.loc[day] += day_energy
        
        # Passe die Verteilung basierend auf dem wöchentlichen Muster an
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

def derive_power_ts_from_energy_ts(energy_ts):
    """
    Leitet Leistungszeitreihe aus Energiezeitreihe ab
    """
    power_ts = energy_ts.copy()
    
    # Sortiere nach Zeitstempel
    power_ts = power_ts.sort_values('timestamp')
    
    # Berechne Leistung als Differenz der Energiewerte
    power_ts['power'] = power_ts['meter_reading'].diff() / power_ts['timestamp'].diff().dt.total_seconds() * 3600  # in kW
    
    # Entferne den ersten Eintrag (keine Differenz möglich)
    power_ts = power_ts.iloc[1:]
    
    return power_ts

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
    
    for complete_day in complete_days:
        # Berechne die drei Distanzmaße
        # 1. Energiedistanz
        de = abs(gap_day_energy - complete_day['energy']) / (e_max - e_min)
        
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

def dd_cpi_imputation(energy_ts, weights=(5, 1, 10)):
    """
    Implementierung der Data-Driven Copy-Paste Imputation nach dem Paper
    
    Parameter:
    - energy_ts: DataFrame mit timestamp, meter_reading und anomaly Spalten
    - weights: Gewichte für die Unähnlichkeitskriterien (Energie, Wochentag, Saison)
    
    Returns:
    - DataFrame mit imputierten Werten
    """
    # Setze timestamp als Index für einfachere Operationen
    energy_ts = energy_ts.set_index('timestamp')
    
    # Schritt 1: Lineare Interpolation einzelner fehlender Werte
    print("Schritt 1: Lineare Interpolation einzelner fehlender Werte...")
    energy_ts = single_value_linear_interpolation(energy_ts)
    
    # Schritt 2: Energieverbrauchsschätzung
    print("Schritt 2: Energieverbrauchsschätzung...")
    daily_energy = calculate_energy_per_day(energy_ts.reset_index())
    non_complete_days = determine_days_with_missing_values(energy_ts.reset_index())
    weekly_pattern = estimate_weekly_pattern_with_prophet(daily_energy, non_complete_days)
    missing_energy = estimate_missing_energy_per_day(energy_ts.reset_index(), weekly_pattern)
    
    # Kombiniere die geschätzte mit der tatsächlichen Energie
    estimated_energy_per_day = daily_energy.copy()
    estimated_energy_per_day.loc[missing_energy.index, 'meter_reading'] += missing_energy
    
    # Schritt 3: Zusammenstellung der verfügbaren vollständigen Tage
    print("Schritt 3: Zusammenstellung der verfügbaren vollständigen Tage...")
    complete_days = compile_list_of_complete_days(
        daily_energy.index, 
        estimated_energy_per_day, 
        non_complete_days
    )
    
    # Schritt 4: Berechne die Leistungszeitreihe
    print("Schritt 4: Ableitung der Leistungszeitreihe...")
    power_ts = derive_power_ts_from_energy_ts(energy_ts.reset_index())
    
    # Schritt 5: Identifiziere Tage mit Lücken und finde beste Übereinstimmungen
    print("Schritt 5: Finde und kopiere die besten passenden Tage...")
    days_with_gaps = []
    for day, is_incomplete in zip(daily_energy.index, non_complete_days):
        if is_incomplete:
            days_with_gaps.append({
                'date': day,
                'energy': estimated_energy_per_day.loc[day, 'meter_reading']
            })
    
    power_ts_imputed = power_ts.copy()
    
    # Kopiere die besten passenden Tage
    for day_with_gaps in days_with_gaps:
        best_matching_day = find_day_with_min_dissimilarity(day_with_gaps, complete_days, weights)
        
        if best_matching_day is None:
            continue
        
        # Kopiere die Leistungswerte des besten passenden Tages
        gap_day_start = day_with_gaps['date']
        gap_day_end = gap_day_start + timedelta(days=1)
        
        match_day_start = best_matching_day['date']
        match_day_end = match_day_start + timedelta(days=1)
        
        # Hole die Leistungswerte des besten passenden Tages
        matching_powers = power_ts[
            (power_ts['timestamp'] >= match_day_start) & 
            (power_ts['timestamp'] < match_day_end)
        ]['power'].values
        
        # Identifiziere die zu ersetzenden Zeitstempel im Tag mit Lücken
        gap_timestamps = power_ts_imputed[
            (power_ts_imputed['timestamp'] >= gap_day_start) & 
            (power_ts_imputed['timestamp'] < gap_day_end) &
            (power_ts_imputed['anomaly'] == 1)
        ]['timestamp'].values
        
        # Falls die Anzahl der Zeitstempel nicht übereinstimmt, passe an
        if len(gap_timestamps) > 0 and len(matching_powers) > 0:
            # Erstelle ein Mapping der Stunden des Tages zu Leistungswerten
            hour_to_power = {}
            for idx, ts in enumerate(power_ts[
                (power_ts['timestamp'] >= match_day_start) & 
                (power_ts['timestamp'] < match_day_end)
            ]['timestamp']):
                hour_key = ts.hour + ts.minute/60
                if idx < len(matching_powers):
                    hour_to_power[hour_key] = matching_powers[idx]
            
            # Ersetze die Leistungswerte im Tag mit Lücken
            for ts in gap_timestamps:
                # Konvertiere numpy.datetime64 zu pandas.Timestamp
                if isinstance(ts, np.datetime64):
                    ts = pd.Timestamp(ts)
                
                hour_key = ts.hour + ts.minute/60
                if hour_key in hour_to_power:
                    power_ts_imputed.loc[power_ts_imputed['timestamp'] == ts, 'power'] = hour_to_power[hour_key]
    
    # Schritt 6: Skaliere die imputierten Werte, um die Energie zu erhalten
    print("Schritt 6: Skaliere die imputierten Werte...")
    # Identifiziere Lücken
    energy_ts_reset = energy_ts.reset_index()
    energy_ts_reset['gap_id'] = (energy_ts_reset['anomaly'].diff() == 1).cumsum()
    gaps = energy_ts_reset[energy_ts_reset['anomaly'] == 1].groupby('gap_id')
    
    for gap_id, gap_data in gaps:
        # Berechne Beginn und Ende der Lücke
        gap_start = gap_data['timestamp'].min()
        gap_end = gap_data['timestamp'].max()
        
        # Finde den Energiewert vor und nach der Lücke
        before_gap_idx = energy_ts_reset[energy_ts_reset['timestamp'] < gap_start]['timestamp'].max()
        after_gap_idx = energy_ts_reset[energy_ts_reset['timestamp'] > gap_end]['timestamp'].min()
        
        if pd.isna(before_gap_idx) or pd.isna(after_gap_idx):
            continue
        
        before_gap_value = energy_ts_reset.loc[energy_ts_reset['timestamp'] == before_gap_idx, 'meter_reading'].values[0]
        after_gap_value = energy_ts_reset.loc[energy_ts_reset['timestamp'] == after_gap_idx, 'meter_reading'].values[0]
        
        # Berechne die tatsächliche Energiedifferenz während der Lücke
        actual_energy = after_gap_value - before_gap_value
        
        # Berechne die imputierte Energie
        imputed_power_values = power_ts_imputed[
            (power_ts_imputed['timestamp'] > gap_start) & 
            (power_ts_imputed['timestamp'] <= gap_end)
        ]['power']
        
        # Die imputierte Energie ist die Summe der Leistungswerte multipliziert mit der Zeitdifferenz (in Stunden)
        time_diff = (gap_end - gap_start).total_seconds() / 3600
        imputed_energy = imputed_power_values.sum() * (time_diff / len(imputed_power_values))
        
        if imputed_energy != 0:
            # Skalierungsfaktor
            scaling_factor = actual_energy / imputed_energy
            
            # Skaliere die imputierten Leistungswerte
            power_ts_imputed.loc[
                (power_ts_imputed['timestamp'] > gap_start) & 
                (power_ts_imputed['timestamp'] <= gap_end),
                'power'
            ] *= scaling_factor
    
    # Schritt 7: Berechne die imputierte Energiezeitreihe aus der imputierten Leistungszeitreihe
    print("Schritt 7: Berechne die imputierte Energiezeitreihe...")
    # Sortiere nach Zeitstempel
    power_ts_imputed = power_ts_imputed.sort_values('timestamp')
    
    # Entferne das 'anomaly'-Flag bei imputierten Werten
    power_ts_imputed.loc[power_ts_imputed['anomaly'] == 1, 'anomaly'] = 0
    
    # Berechne kumulierte Energie aus Leistung
    energy_ts_imputed = power_ts_imputed.copy()
    energy_ts_imputed['meter_reading'] = power_ts_imputed['power'].cumsum() + energy_ts.reset_index().iloc[0]['meter_reading']
    
    return energy_ts_imputed

def visualize_imputation(original_df, imputed_df, window_size=168):
    """
    Visualisiert die originalen und imputierten Daten in einem bestimmten Zeitfenster
    """
    # Finde einen Zeitraum mit Anomalien
    anomaly_indices = original_df[original_df['anomaly'] == 1]['timestamp']
    
    if len(anomaly_indices) > 0:
        center_idx = anomaly_indices.iloc[0]
        start_idx = center_idx - pd.Timedelta(hours=window_size//2)
        end_idx = center_idx + pd.Timedelta(hours=window_size//2)
        
        # Filtere die Daten für das Zeitfenster
        original_window = original_df[(original_df['timestamp'] >= start_idx) & (original_df['timestamp'] <= end_idx)]
        imputed_window = imputed_df[(imputed_df['timestamp'] >= start_idx) & (imputed_df['timestamp'] <= end_idx)]
        
        # Erstelle zwei Subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Energiezeitreihe
        ax1.plot(original_window['timestamp'], original_window['meter_reading'], 'b-', label='Original')
        ax1.plot(imputed_window['timestamp'], imputed_window['meter_reading'], 'g-', label='Imputed')
        
        # Markiere Anomalien
        anomalies = original_window[original_window['anomaly'] == 1]
        ax1.scatter(anomalies['timestamp'], anomalies['meter_reading'], c='r', s=50, label='Anomalies')
        
        ax1.set_title('Original vs. Imputed Energy Readings')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Energy Reading')
        ax1.legend()
        ax1.grid(True)
        
        # Leistungszeitreihe
        if 'power' in original_window.columns and 'power' in imputed_window.columns:
            ax2.plot(original_window['timestamp'], original_window['power'], 'b-', label='Original')
            ax2.plot(imputed_window['timestamp'], imputed_window['power'], 'g-', label='Imputed')
            
            # Markiere Anomalien
            anomalies = original_window[original_window['anomaly'] == 1]
            ax2.scatter(anomalies['timestamp'], anomalies['power'], c='r', s=50, label='Anomalies')
            
            ax2.set_title('Original vs. Imputed Power Readings')
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Power [kW]')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main(file_path):
    print("Lade Daten...")
    df = load_data(file_path)
    print(f"Insgesamt {len(df)} Datenpunkte geladen.")
    print(f"Gefundene Anomalien: {df['anomaly'].sum()}")
    
    print("Führe DD-CPI Imputation durch...")
    imputed_df = dd_cpi_imputation(df)
    
    print("Speichere imputierte Daten...")
    imputed_df.to_csv('imputed_meter_readings_CPI.csv', index=False)
    
    print("Visualisiere die Ergebnisse...")
    visualize_imputation(df, imputed_df)
    
    print("Fertig! Imputierte Daten wurden in 'imputed_meter_readings.csv' gespeichert.")

# Beispielaufruf
if __name__ == "__main__":
    file_path = "id118/Daten/filtered_data_118.csv"  # Passen Sie den Pfad entsprechend an
    main(file_path)