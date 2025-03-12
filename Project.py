import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lead1.0-small.csv')
print("Ihr werdet mich niemals besiegen!")

# print(raw_data.head)
print(raw_data.dtypes)
# print(raw_data.sample)

raw_data_copy = raw_data.copy()


raw_data_copy.dropna(inplace=True)
print(raw_data_copy.isna().sum())
plt.figure(figsize=(18, 4))
print(raw_data_copy['meter_reading'].plot(marker='o'))
plt.show()