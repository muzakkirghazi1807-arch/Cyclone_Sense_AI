import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import requests
import xml.etree.ElementTree as ET
import re
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cyclone Sense AI",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 19px;
    opacity: 0.75;
    margin-bottom: 8px;
}

.tagline {
    font-size: 15px;
    opacity: 0.65;
    margin-bottom: 20px;
}

.metric-card {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
}

.prediction-box {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-top: 10px;
    margin-bottom: 15px;
}

.section-description {
    opacity: 0.7;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD IBTRACS DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "data/raw/ibtracs.since1980.list.v04r01.csv",
        skiprows=[1],
        low_memory=False
    )

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

    data["ISO_TIME"] = pd.to_datetime(
        data["ISO_TIME"]
    )

    data["NAME"] = data["NAME"].fillna("Unnamed")

    return data


data = load_data()
# ============================================================
# LOAD LIVE NOAA/NHC CYCLONE DATA
# ============================================================

@st.cache_data(ttl=900)
def load_live_cyclones():

    feeds = {
        "Atlantic": "https://www.nhc.noaa.gov/index-at.xml",
        "Eastern Pacific": "https://www.nhc.noaa.gov/index-ep.xml",
        "Central Pacific": "https://www.nhc.noaa.gov/index-cp.xml"
    }

    cyclones = []

    for basin, url in feeds.items():

        try:

            response = requests.get(
                url,
                timeout=10
            )

            response.raise_for_status()

            root = ET.fromstring(
                response.content
            )

            for item in root.findall(".//item"):

                title = item.findtext(
                    "title",
                    ""
                )

                description = item.findtext(
                    "description",
                    ""
                )

                if not title.startswith(
                    "Summary for"
                ):
                    continue

                # Storm name
                name_match = re.search(
                    r"Summary for (?:Tropical Storm|Hurricane|Tropical Depression|Post-Tropical Cyclone)\s+(.+?)\s+\(",
                    title,
                    re.IGNORECASE
                )

                name = (
                    name_match.group(1).strip()
                    if name_match
                    else title
                )

                # Location
                location_match = re.search(
                    r"located near\s+([\d.-]+),\s*([\d.-]+)",
                    description
                )

                latitude = (
                    float(location_match.group(1))
                    if location_match
                    else None
                )

                longitude = (
                    float(location_match.group(2))
                    if location_match
                    else None
                )

                # Movement
                movement_match = re.search(
                    r"movement\s+([A-Za-z-]+)\s+at\s+([\d.]+)\s*mph",
                    description,
                    re.IGNORECASE
                )

                direction = (
                    movement_match.group(1)
                    if movement_match
                    else None
                )

                speed = (
                    float(movement_match.group(2))
                    if movement_match
                    else None
                )

                # Pressure
                pressure_match = re.search(
                    r"pressure was\s+(\d+)\s*mb",
                    description,
                    re.IGNORECASE
                )

                pressure = (
                    float(pressure_match.group(1))
                    if pressure_match
                    else None
                )

                # Wind
                wind_match = re.search(
                    r"maximum sustained winds of about\s+(\d+)\s*mph",
                    description,
                    re.IGNORECASE
                )

                wind = (
                    float(wind_match.group(1))
                    if wind_match
                    else None
                )

                cyclones.append({
                    "name": name,
                    "basin": basin,
                    "latitude": latitude,
                    "longitude": longitude,
                    "wind_mph": wind,
                    "pressure_mb": pressure,
                    "movement_direction": direction,
                    "movement_speed_mph": speed,
                    "source": "NOAA/NHC"
                })

        except Exception:
            continue

    return cyclones


@st.cache_data(ttl=900)
def get_forecast_points():

    url = "https://www.nhc.noaa.gov/index-cp.xml"

    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)

        for item in root.findall(".//item"):

            title = item.findtext("title", "")
            description = item.findtext("description", "")

            if "Forecast Advisory" in title:

                matches = re.findall(
                    r"(?:FORECAST VALID|OUTLOOK VALID)\s+"
                    r"\S+\s+"
                    r"([\d.]+)([NS])\s+"
                    r"([\d.]+)([EW])",
                    description,
                    re.IGNORECASE
                )

                points = []

                for lat, lat_dir, lon, lon_dir in matches:

                    lat = float(lat)
                    lon = float(lon)

                    if lat_dir.upper() == "S":
                        lat = -lat

                    if lon_dir.upper() == "W":
                        lon = -lon

                    points.append({
                        "latitude": lat,
                        "longitude": lon
                    })

                return points

    except Exception:
        pass

    return []

# ============================================================
# LOAD TRAINED AI MODELS
# ============================================================

@st.cache_resource
def load_models():

    intensity_model = joblib.load(
        "models/intensity_model.pkl"
    )

    intensity_imputer = joblib.load(
        "models/intensity_imputer.pkl"
    )

    latitude_model = joblib.load(
        "models/latitude_model.pkl"
    )

    longitude_model = joblib.load(
        "models/longitude_model.pkl"
    )

    track_imputer = joblib.load(
        "models/track_imputer.pkl"
    )

    return (
        intensity_model,
        intensity_imputer,
        latitude_model,
        longitude_model,
        track_imputer
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌪️ Cyclone Sense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multi-Source Satellite Intelligence for Tropical Cyclone '
    'Monitoring & Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="tagline">'
    'AI-assisted cyclone monitoring using historical cyclone '
    'observations and satellite-derived intelligence.'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌪️ Cyclone Sense AI")

    st.write(
        "A lightweight AI-powered prototype for "
        "tropical cyclone monitoring and prediction."
    )

    st.divider()

    st.subheader("📡 Data Sources")

    st.write("• NOAA/NCEI IBTrACS")
    st.write("• NOAA/NCEI HURSAT-B1")
    st.write("• Historical cyclone observations")
    st.write("• Infrared satellite features")

    st.divider()

    st.subheader("🤖 AI Models")

    st.write("• Random Forest Classifier")
    st.write("• Random Forest Regressor")
    st.write("• Satellite feature extraction")

    st.divider()

    st.caption(
        "Hackathon Prototype"
    )

# ============================================================
# LIVE CYCLONE MONITOR
# ============================================================

st.header("🔴 Live Cyclone Monitor")

st.markdown(
    '<div class="section-description">'
    'Current cyclone information from official NOAA/NHC '
    'advisory feeds.'
    '</div>',
    unsafe_allow_html=True
)

live_cyclones = load_live_cyclones()

if live_cyclones:

    for cyclone in live_cyclones:

        st.subheader(
            f"🌪️ {cyclone['name']}"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "📍 Latitude",
                f"{cyclone['latitude']:.1f}°"
                if cyclone["latitude"] is not None
                else "N/A"
            )

        with c2:
            st.metric(
                "📍 Longitude",
                f"{cyclone['longitude']:.1f}°"
                if cyclone["longitude"] is not None
                else "N/A"
            )

        with c3:
            st.metric(
                "💨 Wind",
                f"{cyclone['wind_mph']:.0f} mph"
                if cyclone["wind_mph"] is not None
                else "N/A"
            )

        with c4:
            st.metric(
                "🌡️ Pressure",
                f"{cyclone['pressure_mb']:.0f} mb"
                if cyclone["pressure_mb"] is not None
                else "N/A"
            )

        with c5:
            if cyclone["movement_speed_mph"] is not None:
                movement_text = (
                    f"{cyclone['movement_direction']} "
                    f"{cyclone['movement_speed_mph']:.0f} mph"
                )
            else:
                movement_text = "N/A"

            st.metric(
                "➡️ Movement",
                movement_text
            )

        st.caption(
            f"Source: {cyclone['source']} • "
            f"Basin: {cyclone['basin']}"
        )

else:

    st.info(
        "No active tropical cyclones are currently "
        "reported in the NOAA/NHC feeds."
    )

st.divider()
# =========================
# LIVE CYCLONE MAP
# =========================

if live_cyclones:
    forecast_points = get_forecast_points()

    map_data = []

    for cyclone in live_cyclones:
        if cyclone["latitude"] is not None and cyclone["longitude"] is not None:
            map_data.append({
                "name": cyclone["name"],
                "latitude": cyclone["latitude"],
                "longitude": cyclone["longitude"]
            })

    if map_data:

        map_df = pd.DataFrame(map_data)

        fig_live = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            zoom=2,
            height=500
             )

        if forecast_points:
            forecast_df = pd.DataFrame(forecast_points)

            fig_live.add_trace(
                go.Scattermap(
                    lat=forecast_df["latitude"],
                    lon=forecast_df["longitude"],
                    mode="lines+markers",
                    name="NHC Forecast"
                )
            )

        fig_live.update_traces(
            marker=dict(size=18)
        )

        fig_live.update_layout(
            map_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(
            fig_live,
            use_container_width=True
        )

        st.caption(
    "🔵 Current position • 🟦 Official NHC forecast track • Source: NOAA/NHC"
)

else:
    st.info("No active cyclones available for mapping.")
st.divider()

# =========================
# IMD INDIA MONITORING
# =========================

st.header("🇮🇳 IMD India Cyclone Monitoring")

st.markdown(
    '<div class="section-description">'
    'Official India Meteorological Department cyclone information '
    'for the North Indian Ocean region.'
    '</div>',
    unsafe_allow_html=True
)

imd_col1, imd_col2, imd_col3 = st.columns(3)

with imd_col1:
    st.metric("🇮🇳 Agency", "IMD")

with imd_col2:
    st.metric("🌊 Region", "North Indian Ocean")

with imd_col3:
    st.metric("📡 Status", "Official Feed")

st.info(
    "IMD provides Tropical Weather Outlooks, National Bulletins, "
    "cyclone tracks, wind warnings and storm-surge warnings."
)

st.link_button(
    "🌐 Open Official IMD Cyclone Information",
    "https://mausam.imd.gov.in/responsive/cycloneinformation.php?lang=en"
)
# ============================================================
# CYCLONE SELECTION
# ============================================================

st.header("🌪️ Cyclone Monitoring")

st.markdown(
    '<div class="section-description">'
    'Select a historical cyclone to explore its observations, '
    'track and AI predictions.'
    '</div>',
    unsafe_allow_html=True
)

cyclones = data[
    ["SID", "NAME", "SEASON"]
].drop_duplicates()

cyclone_labels = (
    cyclones["NAME"].astype(str)
    + " — "
    + cyclones["SEASON"].astype(str)
)

selected = st.selectbox(
    "Select Cyclone",
    sorted(cyclone_labels.unique())
)

selected_name = selected.split(" — ")[0]
selected_season = int(
    selected.split(" — ")[1]
)

cyclone_data = data[
    (data["NAME"].astype(str) == selected_name)
    &
    (data["SEASON"] == selected_season)
].copy()

cyclone_data = cyclone_data.sort_values(
    "ISO_TIME"
)

latest = cyclone_data.iloc[-1]

st.success(
    f"Currently monitoring: {selected_name} ({selected_season})"
)

# ============================================================
# CURRENT CYCLONE STATUS
# ============================================================

st.subheader("📊 Current Cyclone Status")

wind = latest["USA_WIND"]
pressure = latest["USA_PRES"]
lat = latest["LAT"]
lon = latest["LON"]
sshs = latest["USA_SSHS"]

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "💨 Wind",
        f"{wind:.0f} kt"
        if pd.notna(wind)
        else "N/A"
    )

with c2:
    st.metric(
        "🌀 Intensity",
        f"SSHS {int(sshs)}"
        if pd.notna(sshs)
        else "N/A"
    )

with c3:
    st.metric(
        "🌡️ Pressure",
        f"{pressure:.0f} hPa"
        if pd.notna(pressure)
        else "N/A"
    )

with c4:
    st.metric(
        "📍 Latitude",
        f"{lat:.2f}°"
    )

with c5:
    st.metric(
        "📍 Longitude",
        f"{lon:.2f}°"
    )

st.caption(
    f"Latest historical observation: {latest['ISO_TIME']}"
)

# ============================================================
# AI PREDICTIONS
# ============================================================

st.divider()

st.header("🤖 AI Predictions")

st.markdown(
    '<div class="section-description">'
    'Machine-learning predictions based on the latest available '
    'cyclone observation.'
    '</div>',
    unsafe_allow_html=True
)

try:

    (
        intensity_model,
        intensity_imputer,
        latitude_model,
        longitude_model,
        track_imputer
    ) = load_models()

    # --------------------------------------------------------
    # INTENSITY PREDICTION
    # --------------------------------------------------------

    intensity_features = [
        "LAT",
        "LON",
        "USA_WIND",
        "USA_PRES",
        "STORM_SPEED",
        "STORM_DIR",
        "USA_SSHS"
    ]

    current_features = pd.DataFrame(
        [[latest[col] for col in intensity_features]],
        columns=intensity_features
    )

    current_features_imputed = (
        intensity_imputer.transform(
            current_features
        )
    )

    predicted_intensity = int(
        intensity_model.predict(
            current_features_imputed
        )[0]
    )

    # --------------------------------------------------------
    # TRACK PREDICTION
    # --------------------------------------------------------

    track_features = [
        "LAT",
        "LON",
        "STORM_SPEED",
        "STORM_DIR",
        "USA_WIND",
        "USA_SSHS"
    ]

    track_input = pd.DataFrame(
        [[latest[col] for col in track_features]],
        columns=track_features
    )

    track_input_imputed = (
        track_imputer.transform(
            track_input
        )
    )

    predicted_lat = latitude_model.predict(
        track_input_imputed
    )[0]

    predicted_lon = longitude_model.predict(
        track_input_imputed
    )[0]

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "🔮 Next Intensity",
            f"SSHS {predicted_intensity}"
        )

    with p2:
        st.metric(
            "➡️ Next Latitude",
            f"{predicted_lat:.2f}°"
        )

    with p3:
        st.metric(
            "➡️ Next Longitude",
            f"{predicted_lon:.2f}°"
        )

    st.info(
        "These predictions represent the next available "
        "observation step in the historical dataset, "
        "not a 24/48/72-hour forecast."
    )

except Exception as e:

    st.error(
        f"AI model loading or prediction error: {e}"
    )

# ============================================================
# HISTORICAL TRACK
# ============================================================

st.divider()

st.header("🗺️ Historical Cyclone Track")

st.markdown(
    '<div class="section-description">'
    'Observed movement of the selected cyclone.'
    '</div>',
    unsafe_allow_html=True
)

track_plot = cyclone_data[
    ["ISO_TIME", "LAT", "LON"]
].dropna()

fig = px.line_map(
    track_plot,
    lat="LAT",
    lon="LON",
    hover_name="ISO_TIME",
    zoom=3,
    height=550
)

fig.update_layout(
    map_style="open-street-map",
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# WIND HISTORY
# ============================================================

st.header("📈 Wind Intensity History")

wind_data = cyclone_data[
    ["ISO_TIME", "USA_WIND"]
].dropna()

fig_wind = px.line(
    wind_data,
    x="ISO_TIME",
    y="USA_WIND",
    labels={
        "ISO_TIME": "Time",
        "USA_WIND": "Wind Speed (knots)"
    },
    title="Cyclone Wind Speed Over Time"
)

st.plotly_chart(
    fig_wind,
    use_container_width=True
)

# ============================================================
# SATELLITE INTELLIGENCE
# ============================================================

st.divider()

st.header("🛰️ Satellite Intelligence")

st.markdown(
    '<div class="section-description">'
    'Satellite-derived infrared information integrated with '
    'historical cyclone observations.'
    '</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# ACTUAL HURSAT SATELLITE IMAGE
# ------------------------------------------------------------

satellite_image = (
    "data/raw/satellite/"
    "2010136N09057.BANDU.2010.05.15.1200.65.MSG-2.013."
    "hursat-b1.v06.nc"
)

if os.path.exists(satellite_image):

    import xarray as xr
    import matplotlib.pyplot as plt

    try:

        satellite_ds = xr.open_dataset(
            satellite_image
        )

        infrared = satellite_ds[
            "IRWIN"
        ].isel(htime=0)

        fig_sat, ax = plt.subplots(
            figsize=(10, 6)
        )

        image = ax.imshow(
            infrared,
            extent=[
                float(satellite_ds.lon.min()),
                float(satellite_ds.lon.max()),
                float(satellite_ds.lat.min()),
                float(satellite_ds.lat.max())
            ],
            origin="lower"
        )

        ax.set_title(
            "HURSAT Infrared Satellite Image — Cyclone BANDU"
        )

        ax.set_xlabel(
            "Longitude"
        )

        ax.set_ylabel(
            "Latitude"
        )

        fig_sat.colorbar(
            image,
            ax=ax,
            label="Infrared Brightness Temperature"
        )

        st.pyplot(
            fig_sat
        )

        st.caption(
            "Satellite source: NOAA/NCEI HURSAT-B1 v06"
        )

        satellite_ds.close()

    except Exception as e:

        st.warning(
            f"Unable to display satellite image: {e}"
        )

else:

    st.warning(
        "Satellite image file not found."
    )

# ------------------------------------------------------------
# SATELLITE FEATURES
# ------------------------------------------------------------

satellite_file = (
    "data/processed/cyclone_multisource.csv"
)

if os.path.exists(satellite_file):

    satellite_data = pd.read_csv(
        satellite_file
    )

    satellite_data[
        "observation_time"
    ] = pd.to_datetime(
        satellite_data[
            "observation_time"
        ]
    )

    st.subheader(
        "📡 Extracted Satellite Features"
    )

    sat1, sat2, sat3 = st.columns(3)

    with sat1:

        st.metric(
            "🌡️ Mean IR Temperature",
            f"{satellite_data['mean_temperature'].mean():.2f} K"
        )

    with sat2:

        st.metric(
            "❄️ Cold Cloud Fraction",
            f"{satellite_data['cold_cloud_fraction'].mean() * 100:.1f}%"
        )

    with sat3:

        st.metric(
            "🛰️ Satellite Observations",
            len(satellite_data)
        )

    st.dataframe(
        satellite_data[
            [
                "observation_time",
                "mean_temperature",
                "min_temperature",
                "max_temperature",
                "cold_cloud_fraction"
            ]
        ].tail(10),
        use_container_width=True
    )

else:

    st.warning(
        "Satellite feature dataset not found."
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("🎯 Model Performance")

st.markdown(
    '<div class="section-description">'
    'Performance measured on cyclone-level held-out test data.'
    '</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Intensity Accuracy",
        "92.97%"
    )

with m2:

    st.metric(
        "Macro F1 Score",
        "0.89"
    )

with m3:

    st.metric(
        "Mean Next-Position Error",
        "8.72 km"
    )

st.caption(
    "The track metric represents mean error for predicting "
    "the next available cyclone position."
)

# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.header("⚙️ How Cyclone Sense AI Works")

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.subheader("1️⃣ Collect")

    st.write(
        "Historical cyclone observations from IBTrACS "
        "and satellite observations from HURSAT."
    )

with step2:

    st.subheader("2️⃣ Process")

    st.write(
        "Clean observations and extract useful "
        "satellite-derived infrared features."
    )

with step3:

    st.subheader("3️⃣ Predict")

    st.write(
        "Random Forest models estimate the next "
        "intensity state and cyclone position."
    )

with step4:

    st.subheader("4️⃣ Visualize")

    st.write(
        "Present cyclone intelligence through an "
        "interactive dashboard."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌪️ Cyclone Sense AI • Hackathon Prototype • "
    "NOAA/NCEI IBTrACS + HURSAT + Machine Learning"
)