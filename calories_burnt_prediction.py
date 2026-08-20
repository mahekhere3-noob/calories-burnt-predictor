import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
# Expected columns: Gender, Age, Weight, Duration, Heart_Rate, Body_Temp, Calories_Burnt
data = pd.read_csv("calories_burnt_data.csv")

# Convert Categorical Data
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])

# Features and Target
X = data[["Gender", "Age", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
y = data["Calories_Burnt"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Test Accuracy
prediction = model.predict(X_test)

print("Model R² Score:", r2_score(y_test, prediction))
print("Mean Absolute Error:", mean_absolute_error(y_test, prediction), "calories")

# User Prediction
gender = input("Enter Gender (Male/Female): ")
age = float(input("Enter Age: "))
weight = float(input("Enter Weight (kg): "))
duration = float(input("Enter Workout Duration (minutes): "))
heart_rate = float(input("Enter Average Heart Rate (bpm): "))
body_temp = float(input("Enter Body Temperature (°C): "))

gender_enc = le.transform([gender])[0]

new_data = [[gender_enc, age, weight, duration, heart_rate, body_temp]]

result = model.predict(new_data)

print(f"\nPredicted Calories Burnt: {result[0]:.1f} kcal")
