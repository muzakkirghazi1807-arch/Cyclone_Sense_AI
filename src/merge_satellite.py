import pandas as pd

# ==============================
# LOAD IBTRACS
# ==============================

ibtracs = pd.read_csv(
    "data/raw/ibtracs.since1980.list.v04r01.csv",
    skiprows=[1],
    low_memory=False
)

# Convert important columns
ibtracs["ISO_TIME"] = pd.to_datetime(ibtracs["ISO_TIME"])

numeric_columns = [
    "LAT",
    "LON",
    "USA_WIND",
    "USA_PRES",
    "USA_SSHS"
]

for column in numeric_columns:
    ibtracs[column] = pd.to_numeric(
        ibtracs[column],
        errors="coerce"
    )


# ==============================
# LOAD SATELLITE FEATURES
# ==============================

satellite = pd.read_csv(
    "data/processed/satellite_features.csv"
)

satellite["observation_time"] = pd.to_datetime(
    satellite["observation_time"]
)

# Remove .BANDU from satellite cyclone ID
satellite["SID"] = satellite["cyclone_id"].str.split(".").str[0]


# ==============================
# PREPARE TIMES
# ==============================

# HURSAT timestamps contain tiny floating-point
# differences such as:
# 11:59:59.999959808
#
# Round both datasets to the nearest hour.

ibtracs["match_time"] = ibtracs["ISO_TIME"].dt.round("h")

satellite["match_time"] = satellite[
    "observation_time"
].dt.round("h")


# ==============================
# MERGE
# ==============================

merged = pd.merge(
    satellite,
    ibtracs[
        [
            "SID",
            "match_time",
            "ISO_TIME",
            "LAT",
            "LON",
            "USA_WIND",
            "USA_PRES",
            "USA_SSHS"
        ]
    ],
    on=["SID", "match_time"],
    how="left"
)

# ==============================
# SAVE
# ==============================

output_path = "data/processed/satellite_ibtracs.csv"

merged.to_csv(
    output_path,
    index=False
)


# ==============================
# RESULTS
# ==============================

print("=" * 60)
print("🛰️ SATELLITE + IBTRACS MERGE")
print("=" * 60)

print(f"\nSatellite records : {len(satellite)}")
print(f"Merged records    : {len(merged)}")

matched = merged["LAT"].notna().sum()

print(f"Matched records   : {matched}")
print(
    f"Match rate        : "
    f"{matched / len(merged):.2%}"
)

print("\nSample merged data:")

print(
    merged[
        [
            "cyclone_name",
            "observation_time",
            "mean_temperature",
            "cold_cloud_fraction",
            "LAT",
            "LON",
            "USA_WIND",
            "USA_SSHS"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved to:")
print(output_path)