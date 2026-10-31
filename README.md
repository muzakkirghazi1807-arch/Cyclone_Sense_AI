# 🌪️ Cyclone Sense AI

### Multi-Source Satellite Intelligence for Tropical Cyclone Monitoring, Classification and Prediction

Cyclone Sense AI is an AI-powered platform designed to bring historical cyclone data, satellite observations, machine-learning predictions, and official weather information together in one interactive dashboard.

The project aims to make cyclone information easier to monitor, explore, and understand through data visualization and AI-based analysis.

## 🚀 Key Features

* **Cyclone Monitoring:** Displays available cyclone information from official weather sources.
* **AI-Based Intensity Estimation:** Uses a machine-learning model to estimate cyclone intensity.
* **Cyclone Track Prediction:** Uses historical cyclone observations to estimate the next recorded position.
* **Historical Data Analysis:** Explores cyclone records from the IBTrACS dataset.
* **Satellite Data Analysis:** Uses historical HURSAT-B1 satellite data for cyclone-related analysis.
* **Official Weather Information:** Integrates available NOAA/NHC information and India Meteorological Department (IMD) bulletins.
* **Interactive Dashboard:** Presents information through maps, metrics, and visualizations.

## 🧠 How It Works

The platform follows a multi-source data-processing workflow:

1. **Data Collection:** Collects historical cyclone records, satellite data, and available official weather information.
2. **Data Processing:** Cleans, prepares, and combines relevant datasets.
3. **AI Analysis:** Applies machine-learning models to estimate cyclone intensity and predict the next recorded position.
4. **Visualization:** Displays the processed information and predictions through an interactive dashboard.

## 🛠️ Technology Stack

| Technology     | Purpose                             |
| -------------- | ----------------------------------- |
| Python         | Core programming and backend logic  |
| Streamlit      | Interactive dashboard               |
| Scikit-learn   | Machine-learning models             |
| Pandas         | Data processing                     |
| NumPy          | Numerical operations                |
| Plotly         | Interactive visualizations          |
| OpenCV         | Image processing                    |
| Git and GitHub | Version control and project hosting |

## 📊 Data Sources

* **IBTrACS:** Historical tropical cyclone best-track records.
* **HURSAT-B1:** Historical satellite observations for tropical cyclone analysis.
* **NOAA/NHC:** Official cyclone information and available monitoring feeds.
* **IMD:** Official Indian meteorological bulletins and weather information.

## 📈 Model Evaluation

The current experimental model evaluation includes:

* Intensity classification accuracy: **92.97%**
* Macro F1-score: **0.89**
* Mean next-position prediction error: **8.72 km**

These results are based on historical test data. They should not be interpreted as guaranteed operational forecasting accuracy.

## 💻 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/muzakkirghazi1807-arch/Cyclone_Sense_AI.git
```

### 2. Open the project folder

```bash
cd Cyclone_Sense_AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

Install the required Python packages used by the project:

```bash
pip install streamlit pandas numpy scikit-learn plotly opencv-python matplotlib scipy requests beautifulsoup4
```

### 5. Prepare the required data and model files

Large raw datasets and trained model files are excluded from this GitHub repository because of their size.

The required datasets and model files must be obtained separately and placed in their corresponding project directories before running features that depend on them.

### 6. Run the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard should open in your browser.

## ⚠️ Current Limitations

* Some large datasets and trained model files are not included in the public repository.
* HURSAT-B1 provides historical satellite observations, not live satellite imagery.
* NOAA/NHC live information covers selected monitoring regions and does not represent complete global coverage.
* The current track model estimates the next recorded historical position; it is not an operational 24-hour, 48-hour, or 72-hour forecasting system.
* Predictions are experimental and should not be used as a substitute for official meteorological warnings.

## 🔮 Future Improvements

* Integration of additional live satellite data sources.
* Improved cyclone detection and classification models.
* More advanced cyclone-track forecasting.
* Better data fusion across satellite and meteorological sources.
* Expanded global cyclone coverage.
* Improved dashboard accessibility and visualization.

## 🎯 Project Goal

Our goal is to develop a unified cyclone intelligence platform that combines multiple data sources and AI-based analysis to support cyclone monitoring, research, and awareness.

---

**Project:** Cyclone Sense AI
**Category:** Artificial Intelligence / Climate Technology
**Repository:** https://github.com/muzakkirghazi1807-arch/Cyclone_Sense_AI
