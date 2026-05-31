# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# =========================================================
# 2. DATA COLLECTION (API Open-Meteo)
# =========================================================
cities = {
    "Marrakech":   {"lat": 31.63, "lon": -8.00},
    "Casablanca":  {"lat": 33.59, "lon": -7.62},
    "Rabat":       {"lat": 34.01, "lon": -6.83},
    "Fes":         {"lat": 34.04, "lon": -5.00},
    "Agadir":      {"lat": 30.42, "lon": -9.60},
    "Tangier":     {"lat": 35.77, "lon": -5.80},
    "Oujda":       {"lat": 34.68, "lon": -1.91},
    "Meknes":      {"lat": 33.89, "lon": -5.56},
    "Laayoune":    {"lat": 27.15, "lon": -13.20},
}

data_list = []

for city, coords in cities.items():
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={coords['lat']}&longitude={coords['lon']}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,surface_pressure,cloudcover,weathercode"
        f"&past_days=30"
    )

    response = requests.get(url)
    data = response.json()

    hourly = data["hourly"]
    n = len(hourly["time"])

    for i in range(n):
        data_list.append({
            "city": city,
            "datetime": hourly["time"][i],
            "temperature": hourly["temperature_2m"][i],
            "humidity": hourly["relativehumidity_2m"][i],
            "wind_speed": hourly["windspeed_10m"][i],
            "surface_pressure": hourly["surface_pressure"][i],
            "cloud_cover": hourly["cloudcover"][i],
            "weathercode": hourly["weathercode"][i],
        })

df = pd.DataFrame(data_list)
print("Data collected:", df.shape)


df_raw = pd.DataFrame(data_list)
df_raw.to_csv("weather_raw.csv", index=False)
print("Raw CSV saved successfully!")

# =========================================================
# 3. DATA CLEANING
# =========================================================
df = df.fillna(method="bfill").fillna(method="ffill")
df = df.drop_duplicates()
df["datetime"] = pd.to_datetime(df["datetime"])

df = df[(df["humidity"] >= 0) & (df["humidity"] <= 100)]
df = df[(df["temperature"] > -30) & (df["temperature"] < 55)]
df = df[(df["wind_speed"] >= 0)]
df = df[(df["cloud_cover"] >= 0) & (df["cloud_cover"] <= 100)]
df = df[(df["surface_pressure"] > 800) & (df["surface_pressure"] < 1100)]

num_cols = ["temperature", "humidity", "wind_speed", "surface_pressure", "cloud_cover"]
z_scores = np.abs(zscore(df[num_cols]))
df = df[(z_scores < 3).all(axis=1)]

print("\nData shape after cleaning:", df.shape)

df.to_csv("weather_data_cleaned.csv", index=False)
print("Cleaned CSV saved successfully!")

# =========================================================
# 4. TARGET ENGINEERING (3 classes)
# =========================================================
def classify_weather(code):
    if code <= 2:
        return 0
    elif code <= 49:
        return 1
    else:
        return 2

df["target"] = df["weathercode"].apply(classify_weather)

print("\nTarget distribution:")
print(df["target"].value_counts())

# =========================================================
# 5. FEATURES ENGINEERING + SCALING
# =========================================================
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month

features = ["temperature", "humidity", "wind_speed", "surface_pressure", "cloud_cover", "hour", "month"]

X = df[features]
y = df["target"]

# 5.1 SAVE MODEL DATASET
df_model = df.copy()
df_model.to_csv("weather_model_dataset.csv", index=False)
print("Model dataset CSV saved successfully!")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5.2 SAVE TRAIN / TEST DATA
train_df = pd.DataFrame(X_train, columns=features)
train_df["target"] = y_train.values

test_df = pd.DataFrame(X_test, columns=features)
test_df["target"] = y_test.values

train_df.to_csv("weather_train.csv", index=False)
test_df.to_csv("weather_test.csv", index=False)

print("Train/Test CSV saved successfully!")

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(weights))
print("\nCalculated Class Weights for ANN:", class_weights_dict)

# =========================================================
# 6. EDA
# =========================================================
print(df.describe())

sns.histplot(df["temperature"], kde=True)
plt.title("Temperature Distribution")
plt.show()

sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap with New Features")
plt.show()

# =========================================================
# 7. MACHINE LEARNING (Logistic Regression)
# =========================================================
model_ml = LogisticRegression(max_iter=500, class_weight="balanced")

model_ml.fit(X_train, y_train)
y_pred_ml = model_ml.predict(X_test)

print("\n=== ML RESULTS ===")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_ml))
print(metrics.classification_report(y_test, y_pred_ml))

cm = metrics.confusion_matrix(y_test, y_pred_ml)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — ML")
plt.show()

# =========================================================
# 8. ANN MODEL
# =========================================================
model_ann = Sequential()
model_ann.add(Dense(16, activation="relu", input_shape=(X_train.shape[1],)))
model_ann.add(Dense(8, activation="relu"))
model_ann.add(Dense(3, activation="softmax"))

model_ann.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model_ann.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    verbose=1
)

loss, acc = model_ann.evaluate(X_test, y_test)

print("\n=== ANN RESULTS ===")
print("Accuracy:", acc)

# =========================================================
# ANN CONFUSION MATRIX
# =========================================================
y_pred_ann = np.argmax(model_ann.predict(X_test), axis=1)
cm_ann = metrics.confusion_matrix(y_test, y_pred_ann)

sns.heatmap(cm_ann, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix — ANN")
plt.show()

# =========================================================
# 9. LEARNING CURVES
# =========================================================
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("ANN Loss Curve")
plt.show()

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("ANN Accuracy Curve")
plt.show()

# =========================================================
# 10. FINAL COMPARISON
# =========================================================
ml_acc = metrics.accuracy_score(y_test, y_pred_ml)

print("\n=== FINAL COMPARISON ===")
print("ML Accuracy :", ml_acc)
print("ANN Accuracy:", acc)

if acc > ml_acc:
    print("ANN performs better")
else:
    print("ML performs better")