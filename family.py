import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('family-data.csv')

df['relationship'] = df['relationship'].fillna('Unknown')

# How two numbers are related?
plt.scatter(df['height_inches'], df['weight_lbs'])
plt.xlabel("Height (inches)")
plt.ylabel("Weight (lbs)")
plt.title("Weight vs Height")
plt.show()

# How is one number distributed?
''' plt.hist(df['age'], bins=5, edgecolor='black')
plt.xlabel("Age (Years)")
plt.ylabel("Number of People")
plt.title("Age Distribution")
plt.show() '''

# How many per catergory?
''' relationship_counts = df['relationship'].value_counts()
plt.bar(relationship_counts.index,relationship_counts.values)
plt.xlabel('Relationships')
plt.ylabel("Count")
plt.title("Family Relations")
plt.xticks(rotation=45)
plt.show() '''
