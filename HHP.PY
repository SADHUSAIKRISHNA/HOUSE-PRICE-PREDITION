import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Step 1: Create sample dataset
data = {
    'Area_sqft': [800, 900, 1000, 1100, 1200, 1300, 1500, 1600, 1800, 2000],
    'Rooms': [2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
    'Location_Score': [3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
    'Price_Lakhs': [30, 35, 40, 45, 50, 55, 65, 70, 80, 90]
}
df = pd.DataFrame(data)
# Step 2: Exploratory Data Analysis
print(df.head())
# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()
# Step 3: Feature Selection
X = df[['Area_sqft', 'Rooms', 'Location_Score']]
y = df['Price_Lakhs']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Step 5: Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
# Step 6: Predictions

y_pred = model.predict(X_test)


# Step 7: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
# Step 8: Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance:")
print(importance)
