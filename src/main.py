import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report


# ============================================
# 1. LOAD DATA
# ============================================

file_path = "data/raw/ibtracs.since1980.list.v04r01.csv"

data = pd.read_csv(
    file_path,
    skiprows=[1],
    low_memory=False
)

print("✅ Dataset loaded!")


# ============================================
# 2. CONVERT NUMERIC COLUMNS
# ============================================

numeric_columns = [
    "LAT",
    "LON",
    "USA_WIND",
    "USA_PRES",
    "USA_SSHS",
    "STORM_SPEED",
    "STORM_DIR"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================
# 3. CONVERT TIME
# ============================================

data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"])


# ============================================
# 4. SORT BY CYCLONE AND TIME
# ============================================

data = data.sort_values(
    ["SID", "ISO_TIME"]
).reset_index(drop=True)


# ============================================
# 5. CREATE FUTURE TARGETS
# ============================================

# Next intensity category
data["FUTURE_SSHS"] = (
    data.groupby("SID")["USA_SSHS"]
    .shift(-1)
)

# Next cyclone position
data["NEXT_LAT"] = (
    data.groupby("SID")["LAT"]
    .shift(-1)
)

data["NEXT_LON"] = (
    data.groupby("SID")["LON"]
    .shift(-1)
)

# ============================================
# 6. REMOVE ROWS WITH NO FUTURE TARGET
# ============================================

data = data.dropna(
    subset=["FUTURE_SSHS", "NEXT_LAT", "NEXT_LON"]
).copy()
# ============================================
# 7. SELECT FEATURES
# ============================================

features = [
    "LAT",
    "LON",
    "USA_WIND",
    "USA_PRES",
    "STORM_SPEED",
    "STORM_DIR",
    "USA_SSHS"
]

X = data[features]

y = data["FUTURE_SSHS"].astype(int)


print("\n📊 Machine Learning Dataset:")
print("Features:", X.shape)
print("Target:", y.shape)

print("\n🎯 Target distribution:")
print(y.value_counts().sort_index())


# ============================================
# 8. SPLIT BY CYCLONE
# ============================================

from sklearn.model_selection import GroupShuffleSplit

groups = data["SID"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("\n📚 Training samples:", len(X_train))
print("🧪 Testing samples:", len(X_test))

print(
    "🌪️ Training cyclones:",
    data.iloc[train_index]["SID"].nunique()
)

print(
    "🌪️ Testing cyclones:",
    data.iloc[test_index]["SID"].nunique()
)

print("\n📚 Training samples:", len(X_train))
print("🧪 Testing samples:", len(X_test))


# ============================================
# 9. HANDLE MISSING VALUES
# ============================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)


# ============================================
# 10. CREATE AI MODEL
# ============================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================
# 11. TRAIN MODEL
# ============================================

print("\n🤖 Training Cyclone Sense AI...")

model.fit(X_train, y_train)
predictions = model.predict(X_test)

import joblib

joblib.dump(model, "models/intensity_model.pkl")
joblib.dump(imputer, "models/intensity_imputer.pkl")

print("\nIntensity model saved!")


# ============================================
# 12. TEST MODEL
# ============================================

predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n🎯 Model Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================
# 13. DETAILED RESULTS
# ============================================

print("\n📋 Classification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# ============================================
# 14. CYCLONE TRACK PREDICTION
# ============================================

print("\n" + "=" * 50)
print("🗺️ TRAINING CYCLONE TRACK MODEL")
print("=" * 50)

# Features for track prediction
track_features = [
    "LAT",
    "LON",
    "STORM_SPEED",
    "STORM_DIR",
    "USA_WIND",
    "USA_SSHS"
]

X_track = data[track_features]

y_lat = data["NEXT_LAT"]
y_lon = data["NEXT_LON"]


# Use the SAME cyclone split as before
X_track_train = X_track.iloc[train_index]
X_track_test = X_track.iloc[test_index]

y_lat_train = y_lat.iloc[train_index]
y_lat_test = y_lat.iloc[test_index]

y_lon_train = y_lon.iloc[train_index]
y_lon_test = y_lon.iloc[test_index]


# Handle missing values
track_imputer = SimpleImputer(
    strategy="median"
)

X_track_train = track_imputer.fit_transform(
    X_track_train
)

X_track_test = track_imputer.transform(
    X_track_test
)


# Latitude model
print("\n🤖 Training latitude model...")

lat_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

lat_model.fit(
    X_track_train,
    y_lat_train
)

print("✅ Latitude model trained!")


# Longitude model
print("\n🤖 Training longitude model...")

lon_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

lon_model.fit(
    X_track_train,
    y_lon_train
)

print("✅ Longitude model trained!")


# Save track models
joblib.dump(lat_model, "models/latitude_model.pkl")
joblib.dump(lon_model, "models/longitude_model.pkl")
joblib.dump(track_imputer, "models/track_imputer.pkl")

print("\n✅ Track models saved!")


# Make predictions
pred_lat = lat_model.predict(X_track_test)
pred_lon = lon_model.predict(X_track_test)


print("\n📍 Example track predictions:")

for i in range(10):
    print(
        f"Actual: ({y_lat_test.iloc[i]:.2f}, "
        f"{y_lon_test.iloc[i]:.2f})   "
        f"Predicted: ({pred_lat[i]:.2f}, "
        f"{pred_lon[i]:.2f})"
    )

# ============================================
# 15. CALCULATE TRACK ERROR
# ============================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two geographical
    coordinates in kilometers.
    """

    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius * c


# Calculate error for every test prediction
track_errors = haversine_distance(
    y_lat_test.values,
    y_lon_test.values,
    pred_lat,
    pred_lon
)


print("\n" + "=" * 50)
print("📏 TRACK PREDICTION PERFORMANCE")
print("=" * 50)

print(
    f"\n📍 Mean Track Error: "
    f"{track_errors.mean():.2f} km"
)

print(
    f"📍 Median Track Error: "
    f"{np.median(track_errors):.2f} km"
)

print(
    f"📍 Minimum Error: "
    f"{track_errors.min():.2f} km"
)

print(
    f"📍 Maximum Error: "
    f"{track_errors.max():.2f} km"
)