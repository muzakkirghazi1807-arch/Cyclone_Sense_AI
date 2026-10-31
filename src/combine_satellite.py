import pandas as pd

# Load merged satellite + IBTrACS data
data = pd.read_csv(
    "data/processed/satellite_ibtracs.csv"
)

# Convert observation time
data["observation_time"] = pd.to_datetime(
    data["observation_time"]
)

# Combine multiple satellite observations
# from the same cyclone and time
combined = (
    data.groupby(
        ["cyclone_id", "cyclone_name", "observation_time"],
        as_index=False
    )
    .agg({
        "mean_temperature": "mean",
        "min_temperature": "mean",
        "max_temperature": "mean",
        "std_temperature": "mean",
        "cold_cloud_fraction": "mean",
        "LAT": "first",
        "LON": "first",
        "USA_WIND": "first",
        "USA_PRES": "first",
        "USA_SSHS": "first"
    })
)

# Sort chronologically
combined = combined.sort_values(
    ["cyclone_id", "observation_time"]
)

# Save
output_path = "data/processed/cyclone_multisource.csv"

combined.to_csv(
    output_path,
    index=False
)

print("=" * 60)
print("🌪️ MULTI-SOURCE DATASET")
print("=" * 60)

print(f"\nOriginal satellite records : {len(data)}")
print(f"Combined records           : {len(combined)}")

print("\nSample:")
print(combined.head(10).to_string(index=False))

print("\nSaved to:")
print(output_path)