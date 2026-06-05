import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('family-data.csv')

df['relationship'] = df['relationship'].fillna('Unknown')

print(df)
