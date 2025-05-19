import pandas as pd

def extract_building_metadata(metadata_file, building_ids, output_file="building_metadata_extract.csv"):
    """
    Extrahiert Metadaten für spezifische Building-IDs aus einer Metadaten-Datei
    und speichert sie in einer neuen Datei.
    
    Args:
        metadata_file (str): Pfad zur Metadaten-Datei (CSV, Excel)
        building_ids (list): Liste der Building-IDs, für die Metadaten extrahiert werden sollen
        output_file (str): Name der Ausgabedatei (Standard: "building_metadata_extract.csv")
    
    Returns:
        str: Pfad zur erstellten Ausgabedatei
    """
    print(f"Extrahiere Metadaten für {len(building_ids)} Buildings...")
    
    # Dateityp der Metadaten-Datei ermitteln
    file_extension = metadata_file.split('.')[-1].lower()
    
    # Metadaten einlesen, basierend auf dem Dateityp
    if file_extension == 'csv':
        metadata_df = pd.read_csv(metadata_file)
    elif file_extension in ['xls', 'xlsx']:
        metadata_df = pd.read_excel(metadata_file)
    else:
        raise ValueError(f"Nicht unterstütztes Dateiformat: {file_extension}. Bitte verwende CSV oder Excel.")
    
    # Sicherstellen, dass building_id als Spalte existiert
    id_column = None
    for col in metadata_df.columns:
        if col.lower() in ['building_id', 'buildingid', 'building', 'id']:
            id_column = col
            break
    
    if id_column is None:
        print("Warnung: Keine Spalte mit Building-ID gefunden. Verwende die erste Spalte.")
        id_column = metadata_df.columns[0]
    
    print(f"Verwende '{id_column}' als ID-Spalte.")
    
    # Konvertiere IDs zu gleichem Datentyp für den Vergleich
    building_ids = [int(id) if isinstance(id, str) and id.isdigit() else id for id in building_ids]
    
    if metadata_df[id_column].dtype != type(building_ids[0]):
        if all(isinstance(id, int) for id in building_ids):
            try:
                metadata_df[id_column] = metadata_df[id_column].astype(int)
            except:
                building_ids = [str(id) for id in building_ids]
                metadata_df[id_column] = metadata_df[id_column].astype(str)
    
    # Filtern der Metadaten nach den angegebenen Building-IDs
    filtered_metadata = metadata_df[metadata_df[id_column].isin(building_ids)]
    
    if filtered_metadata.empty:
        print(f"Keine Metadaten für die angegebenen Building-IDs gefunden.")
        return None
    
    print(f"Gefunden: {len(filtered_metadata)} von {len(building_ids)} Buildings")
    
    # Nicht gefundene Building-IDs auflisten
    found_ids = set(filtered_metadata[id_column].tolist())
    not_found = set(building_ids) - found_ids
    if not_found:
        print(f"Building-IDs ohne Metadaten: {not_found}")
    
    # Ausgabedatei erstellen
    output_extension = output_file.split('.')[-1].lower()
    
    if output_extension == 'csv':
        filtered_metadata.to_csv(output_file, index=False)
    elif output_extension in ['xls', 'xlsx']:
        filtered_metadata.to_excel(output_file, index=False)
    else:
        # Standard: CSV
        if '.' not in output_file:
            output_file += '.csv'
        filtered_metadata.to_csv(output_file, index=False)
    
    print(f"Metadaten wurden in '{output_file}' gespeichert.")
    return output_file

building_ids = [118,148,254,623657,680,685,732,880,881,882,884,890,894,895,914,919,924,925,931,935942,945,948,950,968,988,1068,1128,1238,1246,1249,1252,1253,1259,1261,1284,1304,]
metadata_file = "building_metadata.csv"
extract_building_metadata(metadata_file, building_ids, "ausgewählte_buildings_metadata.csv")



