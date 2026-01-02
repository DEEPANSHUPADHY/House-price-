import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
'Area': [800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
'Bedrooms': [1, 2, 2, 3, 3, 3, 4, 4],
'Age':[18,30,22,24,26,28,30,32],
'Price': [50, 55, 60, 65, 70, 75, 80, 85]# in lakhs

}


#df = pd.DataFrame(data)
df = pd.read_csv("C:/Users/KIIT/AppData/Local/Temp/38769b13-71e7-49fe-a12e-d503c7f18a32_House Price India.csv.zip.a32/House Price India.csv")
df['Age'] = 2025 - df['Built Year']
X = df[['number of bathrooms', 'living area','Area of the house(excluding basement)','Age']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept (b0):", model.intercept_)
print("Coefficients (b1, b2,b3,b4):", model.coef_)

y_pred = model.predict(X_test)


print("Actual Prices:", y_test.values)
print("Predicted Prices:", y_pred)

from sklearn.metrics import mean_squared_error, r2_score


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("MSE:", mse)
print("R2 Score:", r2)


