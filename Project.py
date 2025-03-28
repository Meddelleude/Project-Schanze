import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lead1.0-small.csv')
#print("Ihr werdet mich niemals besiegen!")

#print(raw_data.head)
#print(raw_data.dtypes)
#print(raw_data.sample)

raw_data_copy = raw_data.copy()


raw_data_copy.dropna(inplace=True)
#print(raw_data_copy.head)
#print(raw_data_copy.isna().sum())
#plt.figure(figsize=(18, 4))
#print(raw_data_copy['meter_reading'].plot(marker='o'))
#plt.show()

def count_building_ids_dict(df):
    
    if 'building_id' not in df.columns:
        raise ValueError("Die Spalte 'building_id' existiert nicht in der Datei.")
    
    id_counts = df['building_id'].value_counts()
    return id_counts

print(count_building_ids_dict(raw_data_copy))

