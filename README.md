# 🔥 Calories Burnt Predictor

Predicts how many calories a workout burns, based on physical profile and exercise readings, using a Random Forest Regressor.

Built as part of a hands-on machine learning mini-project series.

---

## 🔍 Overview

Enter gender, age, weight, workout duration, average heart rate, and body temperature, and the model predicts calories burned during that session.

---

## 📊 Dataset

- 350 synthetic workout records
- Features: `Gender`, `Age`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`
- Target: `Calories_Burnt`

> Generated using the Keytel et al. (2005) heart-rate-based calorie estimation formula as a realistic baseline, with random noise added on top — it approximates real exercise physiology but isn't measured data.

---

## 🤖 Model

```python
RandomForestRegressor(n_estimators=150, random_state=42)
```

**Test performance:** R² ≈ 0.93, MAE ≈ 15 kcal — the model explains the large majority of variation in calorie burn, with predictions typically within about 15 calories.

---

## 🛠️ Tech Stack

Python · pandas · scikit-learn

---

## 🚀 Getting Started

```bash
pip install pandas scikit-learn
python calories_burnt_prediction.py
```

---

## ⚠️ Limitations

- Trained on synthetic data derived from a physiological formula, not real wearable-device measurements
- Doesn't account for factors like fitness level, workout type (cardio vs. strength), or altitude
- Not intended as fitness, medical, or nutritional guidance

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
