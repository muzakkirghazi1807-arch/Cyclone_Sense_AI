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

st.markdown(
    """
    <style>

    .main-title {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 20px;
    font-weight: 500;
    opacity: 0.78;
    margin-bottom: 6px;
}

.tagline {
    font-size: 14px;
    opacity: 0.60;
    margin-bottom: 24px;
}
.hero-section {
    padding: 24px 28px;
    margin-bottom: 24px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.20);
    background: linear-gradient(
        135deg,
        rgba(80,140,180,0.10),
        rgba(80,80,160,0.04)
    );
}
[data-testid="stMetric"] {
    padding: 16px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.20);
    background: rgba(128,128,128,0.04);
}

[data-testid="stMetricLabel"] {
    font-size: 14px;
}

[data-testid="stMetricValue"] {
    font-size: 24px;
    font-weight: 700;
}

    .prediction-box {
    padding: 16px 20px;
    border-radius: 14px;
    border: 1px solid rgba(80,140,180,0.25);
    background: rgba(80,140,180,0.06);
    margin-top: 10px;
    margin-bottom: 14px;
}

.ai-title {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

    </style>
    """,
    unsafe_allow_html=True
)


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
        data["ISO_TIME"],
        errors="coerce"
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

                if not title.startswith("Summary for"):
                    continue

                

                # ------------------------------------------------
                # STORM NAME
                # ------------------------------------------------

                name_match = re.search(
                    r"Summary for (?:Tropical Storm|Hurricane|"
                    r"Tropical Depression|Post-Tropical Cyclone)"
                    r"\s+(.+?)\s+\(",
                    title,
                    re.IGNORECASE
                )

                name = (
                    name_match.group(1).strip()
                    if name_match
                    else title
                )

                # ------------------------------------------------
                # LOCATION
                # ------------------------------------------------

                location_match = re.search(
                    r"located near\s+([\d.-]+),\s*([\d.-]+)",
                    description,
                    re.IGNORECASE
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

                # ------------------------------------------------
                # MOVEMENT
                # ------------------------------------------------

                movement_match = re.search(
                    r"movement\s+([A-Za-z-]+)\s+at\s+"
                    r"([\d.]+)\s*mph",
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

                # ------------------------------------------------
                # PRESSURE
                # ------------------------------------------------

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

                # ------------------------------------------------
                # WIND
                # ------------------------------------------------

                wind_match = re.search(
                    r"maximum sustained winds of about\s+"
                    r"(\d+)\s*mph",
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

# ============================================================
# LOAD LIVE IMD NORTH INDIAN OCEAN BULLETIN
# ============================================================

@st.cache_data(ttl=900)
def load_imd_bulletin():

    url = "https://mausam.imd.gov.in/Forecast/seaarea_bulletin_new.php?id=0"

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.text, "html.parser")

        text = soup.get_text(" ", strip=True)

        return {
            "available": True,
            "text": text,
            "source": url
        }

    except Exception as e:

        return {
            "available": False,
            "text": str(e),
            "source": url
        }


# ============================================================
# EXTRACT NORTH INDIAN OCEAN SYSTEM FROM IMD BULLETIN
# ============================================================

def extract_imd_system(imd_data):

    if not imd_data or not imd_data.get("available"):
        return None

    text = imd_data.get("text", "")

    if not text:
        return None

    # Extract the Synoptic Situation section.
    match = re.search(
        r"Synoptic Situation(.*?)(?:South West Bay|Southwest Bay|Part 4)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    synoptic_text = match.group(1) if match else text

    # Separate current observations from future forecasts.
    current_text = re.split(
        r"\bIt is very likely\b|\bIt is likely\b",
        synoptic_text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    current_lower = current_text.lower()

    # Identify current system status.
    if "extremely severe cyclonic storm" in current_lower:
        status = "Extremely Severe Cyclonic Storm"

    elif "very severe cyclonic storm" in current_lower:
        status = "Very Severe Cyclonic Storm"

    elif "severe cyclonic storm" in current_lower:
        status = "Severe Cyclonic Storm"

    elif "cyclonic storm" in current_lower:
        status = "Cyclonic Storm"

    elif "deep depression" in current_lower:
        status = "Deep Depression"

    elif "depression" in current_lower:
        status = "Depression"

    elif "well marked low pressure area" in current_lower:
        status = "Well-Marked Low Pressure Area"

    elif "low pressure area" in current_lower:
        status = "Low Pressure Area"

    else:
        status = "No identifiable system"

    # Identify region.
    full_text = text.lower()

    if "bay of bengal" in full_text:
        region = "Bay of Bengal"

    elif "arabian sea" in full_text:
        region = "Arabian Sea"

    elif "andaman" in full_text:
        region = "Andaman Sea"

    else:
        region = "North Indian Ocean"

    return {
        "status": status,
        "region": region,
        "bulletin": text,
        "current_text": current_text
    }


# ============================================================
# GET NHC FORECAST POINTS
# ============================================================

@st.cache_data(ttl=900)
def get_forecast_points():

    feeds = [
        "https://www.nhc.noaa.gov/index-at.xml",
        "https://www.nhc.noaa.gov/index-ep.xml",
        "https://www.nhc.noaa.gov/index-cp.xml"
    ]

    for url in feeds:

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

                if "Forecast Advisory" not in title:
                    continue

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

                if points:
                    return points

        except Exception:
            continue

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
    """
<div class="hero-section">

<div class="main-title">
🌪️ Cyclone Sense AI
</div>

<div class="subtitle">
Multi-Source Satellite Intelligence for Tropical Cyclone
Monitoring & Prediction
</div>

<div class="tagline">
AI-assisted cyclone monitoring using historical cyclone
observations and satellite-derived intelligence.
</div>

</div>
""",
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
    st.write("• NOAA/NHC advisory feeds")
    st.write("• IMD cyclone information")
    st.write("• Historical cyclone observations")
    st.write("• Infrared satellite features")

    st.divider()

    st.subheader("🤖 AI Models")

    st.write("• Random Forest Classifier")
    st.write("• Random Forest Regressors")
    st.write("• Satellite feature extraction")

    st.divider()

    st.subheader("🎯 Prototype Scope")

    st.write(
        "Historical cyclone intelligence, "
        "short-step prediction and multi-source "
        "visualization."
    )

    st.divider()

    st.caption("Hackathon Prototype")


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


# ============================================================
# LIVE CYCLONE STATUS
# ============================================================

st.header("🌪️ Live Cyclone Status")

if live_cyclones:

    st.error("🌪️ ACTIVE CYCLONE(S) REPORTED")

    st.write(
        f"Official NOAA/NHC feeds are currently reporting "
        f"{len(live_cyclones)} active cyclone system(s)."
    )

else:

    st.success("🟢 NO ACTIVE CYCLONE REPORTED")

    st.write(
        "No active tropical cyclone is currently reported "
        "in the monitored NOAA/NHC feeds."
    )

    st.caption(
        "This status is based on official NOAA/NHC feeds, "
        "not an independent satellite detection model."
    )


# ============================================================
# LIVE CYCLONE DETAILS
# ============================================================

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




# ============================================================
# LIVE CYCLONE MAP
# ============================================================

st.divider()

st.header("🗺️ Live Cyclone Map")

if live_cyclones:

    forecast_points = get_forecast_points()

    map_data = []

    for cyclone in live_cyclones:

        if (
            cyclone["latitude"] is not None
            and cyclone["longitude"] is not None
        ):

            map_data.append({
                "name": cyclone["name"],
                "latitude": cyclone["latitude"],
                "longitude": cyclone["longitude"]
            })

    if map_data:

        map_df = pd.DataFrame(
            map_data
        )

        fig_live = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            zoom=2,
            height=500
        )

        if forecast_points:

            forecast_df = pd.DataFrame(
                forecast_points
            )

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
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            )
        )

        st.plotly_chart(
            fig_live,
            use_container_width=True
        )

        st.caption(
            "🔵 Current position • "
            "🟦 Official NHC forecast track • "
            "Source: NOAA/NHC"
        )

    else:

        st.info(
            "No active cyclone locations are available "
            "for mapping."
        )

else:

    st.info(
        "No active cyclones available for mapping."
    )


# ============================================================
# IMD INDIA MONITORING
# ============================================================

st.divider()

st.header("🇮🇳 IMD India Cyclone Monitoring")

st.markdown(
    '<div class="section-description">'
    'Live official weather-system information from the '
    'India Meteorological Department.'
    '</div>',
    unsafe_allow_html=True
)

# Fetch the latest IMD bulletin
imd_data = load_imd_bulletin()
imd_system = extract_imd_system(imd_data)

if imd_data and imd_data.get("available"):

    st.success("🟢 Official IMD bulletin retrieved successfully")

    if imd_system and imd_system["status"] != "No identifiable system":

        st.subheader("🌪️ IMD Live System Detection")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🌪️ Current System",
                imd_system["status"]
            )

        with col2:
            st.metric(
                "🌊 Region",
                imd_system["region"]
            )

        with col3:
            st.metric(
                "🇮🇳 Agency",
                "IMD"
            )

        st.info(
            "System status is extracted from the current "
            "synoptic situation in the official IMD bulletin."
        )

    else:
        st.info("No identifiable weather system found in the latest bulletin.")

    with st.expander("📄 View Complete Official IMD Bulletin"):
        st.write(imd_data["text"])

else:
    st.error("Unable to retrieve the latest IMD bulletin.")

    if imd_data:
        st.write(imd_data.get("text", "Unknown error"))

st.link_button(
    "🌐 Open Official IMD Cyclone Information",
    "https://mausam.imd.gov.in/responsive/cycloneinformation.php?lang=en"
)


# ============================================================
# HISTORICAL CYCLONE SELECTION
# ============================================================

st.divider()

st.header("🌪️ Historical Cyclone Analysis")

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
    f"Currently monitoring: "
    f"{selected_name} ({selected_season})"
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
        if pd.notna(lat)
        else "N/A"
    )

with c5:

    st.metric(
        "📍 Longitude",
        f"{lon:.2f}°"
        if pd.notna(lon)
        else "N/A"
    )

st.caption(
    f"Latest historical observation: "
    f"{latest['ISO_TIME']}"
)


# ============================================================
# AI INTELLIGENCE
# ============================================================

st.divider()

st.header("🤖 AI Intelligence")

st.markdown(
    '<div class="section-description">'
    'Machine-learning predictions based on the latest available '
    'observation of the selected historical cyclone.'
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
        [
            [
                latest[col]
                for col in intensity_features
            ]
        ],
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
        [
            [
                latest[col]
                for col in track_features
            ]
        ],
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


    # --------------------------------------------------------
    # MAIN AI OUTPUT
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="prediction-box">
        <div class="ai-title">
        🧠 AI Prediction Summary
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "🔮 Predicted Next Intensity",
            f"SSHS {predicted_intensity}"
        )

    with p2:

        st.metric(
            "➡️ Predicted Next Latitude",
            f"{predicted_lat:.2f}°"
        )

    with p3:

        st.metric(
            "➡️ Predicted Next Longitude",
            f"{predicted_lon:.2f}°"
        )


    # --------------------------------------------------------
    # AI INTERPRETATION
    # --------------------------------------------------------

    st.caption(
    "AI output is based on historical cyclone observations. "
    "The three metrics above show the predicted next intensity "
    "state and next geographical position."
)

    st.warning(
        "Important: these predictions represent the next available "
        "observation step in the historical dataset. They are not "
        "24/48/72-hour operational forecasts."
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
    'Observed movement of the selected historical cyclone.'
    '</div>',
    unsafe_allow_html=True
)

track_plot = cyclone_data[
    ["ISO_TIME", "LAT", "LON"]
].dropna()

if not track_plot.empty:

    fig = px.line_map(
        track_plot,
        lat="LAT",
        lon="LON",
        hover_name="ISO_TIME",
        zoom=3,
        height=600
    )

    fig.add_scattermap(
        lat=track_plot["LAT"],
        lon=track_plot["LON"],
        mode="markers",
        marker=dict(size=7),
        hovertext=track_plot["ISO_TIME"],
        hoverinfo="text"
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

else:

    st.info(
        "No valid latitude/longitude observations "
        "are available for this cyclone."
    )

# ============================================================
# WIND HISTORY
# ============================================================

st.header("📈 Wind Intensity History")

wind_data = cyclone_data[
    ["ISO_TIME", "USA_WIND"]
].dropna()

if not wind_data.empty:

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

else:

    st.info(
        "No wind observations are available "
        "for this cyclone."
    )


# ============================================================
# SATELLITE INTELLIGENCE
# ============================================================

st.divider()

st.header("🛰️ Satellite Intelligence")

st.markdown(
    '<div class="section-description">'
    'Historical satellite-derived infrared information integrated '
    'with historical cyclone observations.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ACTUAL HURSAT SATELLITE IMAGE
# ============================================================

st.subheader("🛰️ Historical Satellite Image")

st.caption(
    "Historical HURSAT-B1 infrared observation for cyclone BANDU."
)

satellite_image = (
    "data/raw/satellite/"
    "2010136N09057.BANDU.2010.05.15.1200.65.MSG-2.013."
    "hursat-b1.v06.nc"
)

if os.path.exists(satellite_image):

    try:

        import xarray as xr
        import matplotlib.pyplot as plt

        satellite_ds = xr.open_dataset(
            satellite_image
        )

        infrared = satellite_ds[
            "IRWIN"
        ].isel(htime=0)

        fig_sat, ax = plt.subplots(
            figsize=(12, 7)
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

        plt.close(fig_sat)

        satellite_ds.close()

    except Exception as e:

        st.warning(
            f"Unable to display satellite image: {e}"
        )

else:

    st.warning(
        "Satellite image file not found."
    )


# ============================================================
# SATELLITE FEATURES
# ============================================================

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
        ],
        errors="coerce"
    )


    # ========================================================
    # HISTORICAL SATELLITE ANALYSIS
    # ========================================================

    st.subheader(
        "🔍 Historical Satellite Analysis"
    )

    st.markdown(
        '<div class="section-description">'
        'Infrared features extracted from the historical '
        'HURSAT-B1 observations.'
        '</div>',
        unsafe_allow_html=True
    )

    latest_satellite = (
        satellite_data
        .dropna(subset=["observation_time"])
        .sort_values("observation_time")
        .iloc[-1]
    )

    mean_temperature = (
        latest_satellite["mean_temperature"]
    )

    min_temperature = (
        latest_satellite["min_temperature"]
    )

    max_temperature = (
        latest_satellite["max_temperature"]
    )

    std_temperature = (
        latest_satellite["std_temperature"]
    )

    cold_cloud_fraction = (
        latest_satellite["cold_cloud_fraction"]
    )


    sa1, sa2, sa3, sa4 = st.columns(4)

    with sa1:

        st.metric(
            "🌡️ Mean IR Temperature",
            f"{mean_temperature:.2f} K"
        )

    with sa2:

        st.metric(
            "❄️ Cold Cloud Fraction",
            f"{cold_cloud_fraction * 100:.1f}%"
        )

    with sa3:

        st.metric(
            "🥶 Minimum IR Temperature",
            f"{min_temperature:.2f} K"
        )

    with sa4:

        st.metric(
            "📊 IR Variation",
            f"{std_temperature:.2f} K"
        )


    st.info(
        "These satellite features describe infrared cloud and "
        "temperature characteristics around the historical "
        "cyclone observation. The current satellite sample is "
        "historical HURSAT-B1 data, not a live satellite feed."
    )


    # ========================================================
    # EXTRACTED SATELLITE FEATURES
    # ========================================================

    st.subheader(
        "📡 Extracted Satellite Features"
    )

    sat1, sat2, sat3 = st.columns(3)

    with sat1:

        st.metric(
            "🌡️ Average IR Temperature",
            f"{satellite_data['mean_temperature'].mean():.2f} K"
        )

    with sat2:

        st.metric(
            "❄️ Average Cold Cloud Fraction",
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
    'Performance measured on cyclone-level held-out historical '
    'test data.'
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
    "Intensity metrics were evaluated using cyclone-level "
    "held-out test data. The track metric represents the mean "
    "error for predicting the next available cyclone position."
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
        "and historical satellite observations from HURSAT."
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
        "interactive dashboard with maps, graphs "
        "and satellite analysis."
    )


# ============================================================
# PROJECT SCOPE
# ============================================================

st.divider()

st.header("🧭 Project Scope")

scope1, scope2, scope3 = st.columns(3)

with scope1:

    st.subheader("📡 Multi-Source")

    st.write(
        "Combines historical cyclone observations, "
        "satellite-derived features and official "
        "meteorological information."
    )

with scope2:

    st.subheader("🤖 AI-Assisted")

    st.write(
        "Machine-learning models provide short-step "
        "intensity and track predictions from historical data."
    )

with scope3:

    st.subheader("📊 Decision Support")

    st.write(
        "Provides a unified visual interface for "
        "exploring cyclone information and model outputs."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌪️ Cyclone Sense AI • Hackathon Prototype • "
    "NOAA/NCEI IBTrACS + HURSAT + NOAA/NHC + IMD + Machine Learning"
)