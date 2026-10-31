import xarray as xr
import numpy as np
import pandas as pd
import glob
import os

# Find all satellite NetCDF files
file_paths = glob.glob("data/raw/satellite/*.nc")

print("=" * 60)
print("🛰️ SATELLITE FEATURE EXTRACTION")
print("=" * 60)

print(f"\nFound {len(file_paths)} satellite files.")

results = []

for i, file_path in enumerate(file_paths):

    try:
        dataset = xr.open_dataset(file_path)

        # Use infrared channel
        image = dataset["IRWIN"].isel(htime=0).values

        # Calculate features
        mean_temperature = np.nanmean(image)
        min_temperature = np.nanmin(image)
        max_temperature = np.nanmax(image)
        std_temperature = np.nanstd(image)

        # Cold cloud fraction
        cold_threshold = 220
        cold_pixels = np.sum(image < cold_threshold)
        total_pixels = np.sum(~np.isnan(image))

        if total_pixels > 0:
            cold_cloud_fraction = cold_pixels / total_pixels
        else:
            cold_cloud_fraction = np.nan

        # Get cyclone information
        cyclone_id = str(dataset.attrs.get("TC_id", "Unknown"))
        cyclone_name = str(dataset.attrs.get("TC_name", "Unknown"))

        # Get satellite observation time
        observation_time = str(dataset.htime.values[0])

        results.append({
            "file": os.path.basename(file_path),
            "cyclone_id": cyclone_id,
            "cyclone_name": cyclone_name,
            "observation_time": observation_time,
            "mean_temperature": mean_temperature,
            "min_temperature": min_temperature,
            "max_temperature": max_temperature,
            "std_temperature": std_temperature,
            "cold_cloud_fraction": cold_cloud_fraction
        })

        print(
            f"[{i + 1}/{len(file_paths)}] "
            f"{cyclone_name} | {observation_time} | "
            f"Cold clouds: {cold_cloud_fraction:.2%}"
        )

        dataset.close()

    except Exception as e:
        print(f"❌ Error processing {file_path}")
        print(e)


# Convert results into a DataFrame
features_df = pd.DataFrame(results)

# Save the feature dataset
output_path = "data/processed/satellite_features.csv"

features_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("✅ SATELLITE FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(f"\nProcessed files: {len(features_df)}")
print(f"Saved to: {output_path}")

print("\nFirst few rows:")
print(features_df.head())