import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('family-data.csv')

df['relationship'] = df['relationship'].fillna('Unknown')

# How two numbers are related?
'''plt.scatter(df['height_inches'], df['weight_lbs'])
plt.xlabel("Height (inches)")
plt.ylabel("Weight (lbs)")
plt.title("Weight vs Height")
plt.show()'''

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


X = df[['height_inches']] # Set X = to 2D datafram of height
y = df['weight_lbs'] # Set y = 1D series of weight

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42) 
''' initalize variables X_train(training height), X_Test(testing height), y_train(training weight), y_test(testing weight), call the 
train_test_split function which splits the two into training sets and testing sets, test-size=(what percent you want to go towards testing), random state
makes sure that every time it is random so the model doesnt just memorize'''

model = LinearRegression() # initalize empty linear regression model

model.fit(X_train, y_train) # take that model and start traning it with training weight and height

y_pred = model.predict(X_test) # after the model is trained use it to predict use the test cases

print("Actual weight:", y_test.values) # actual weights
print("Predicted Weights:", y_pred) # predicted weights

mae = mean_absolute_error(y_test, y_pred) # print the absolute error like if actual weight was 100 and prediciton was 200 mean sbolute errors is diffrence (divded by total outputs)
mse = mean_squared_error(y_test, y_pred) # same as mean asbolute error but penalizes the model more for getting the prediction worst

print(f'Mean Absolute Squared {mae:.2f} lbs') # print 
print(f'Mean Squared Error {mse:.2f}') # print

print(f"Formula: weight = {model.coef_[0]:.2f} * height + {model.intercept_:.2f}")
