import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    'SquareFootage': [1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 1600, 1900],
    'Bedrooms': [2, 3, 3, 4, 4, 4, 5, 5, 3, 3],
    'Bathrooms': [1, 2, 2, 3, 3, 3, 4, 4, 2, 2],
    'Price': [150000, 200000, 240000, 300000, 330000, 380000, 450000, 500000, 220000, 270000]
}

df = pd.DataFrame(data)

X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Prices:", y_test.values)
print("Predicted Prices:", y_pred)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

new_house = [[2100, 4, 3]]
predicted_price = model.predict(new_house)

print("\nPredicted Price for new house:", predicted_price[0])