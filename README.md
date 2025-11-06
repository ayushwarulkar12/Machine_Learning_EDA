# NYC Motor Vehicle Collisions — EDA & Machine Learning

**End-to-end data science project** analyzing the NYC Open Data “Motor Vehicle Collisions — Crashes” dataset. It covers data cleaning, feature engineering, exploratory analysis, supervised models to predict injury likelihood, unsupervised clustering to reveal patterns, time-series forecasting with Prophet, and survival analysis of crash intervals.

> **Author:** Ayush Warulkar
> **Status:** Active • Semester VII Mini Project
> **Stack:** Python • pandas • scikit-learn • matplotlib • seaborn • Prophet • lifelines

---

## Table of Contents

* [Project Goals](#project-goals)
* [Dataset](#dataset) https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes
* [Repository Structure](#repository-structure)
* [Quick Start](#quick-start)
* [Reproducibility](#reproducibility)
* [EDA Highlights](#eda-highlights)
* [Modeling](#modeling)
* [Unsupervised, Forecasting & Survival](#unsupervised-forecasting--survival)
* [Results](#results)
* [Notes & Limitations](#notes--limitations)
* [Roadmap](#roadmap)
* [How to Cite](#how-to-cite)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Project Goals

1. **Describe** temporal and spatial patterns in NYC crashes.
2. **Predict** injury occurrence (binary: any injury vs. none).
3. **Discover** latent clusters of crash behavior.
4. **Forecast** near-term crash trends.
5. **Quantify** time-between-crash dynamics via survival analysis.

---

## Dataset

* **Source:** NYC Open Data — *Motor Vehicle Collisions – Crashes* (CSV).
* **Core fields:** Crash date/time, borough/ZIP, lat/long, vehicle type, contributing factors, injured/killed counts.
* **Target:** `INJURY_FLAG` = 1 if any person injured, else 0.

> Download the dataset from NYC Open Data and place it in `data/` (see structure below). Update the notebook path if needed.

---

## Repository Structure

```
.
├─ data/
│  └─ Motor_Vehicle_Collisions_-_Crashes.csv   # (add locally; not versioned)
├─ notebooks/
│  └─ ML_Motor_Vehicle_Collisions_Crashes.ipynb
├─ reports/
│  └─ ML_Report_Ayush.pdf
├─ src/
│  ├─ preprocessing.py        # helpers for cleaning, feature engineering
│  ├─ modeling.py             # train/evaluate models
│  ├─ clustering.py           # KMeans/DBSCAN/Agglomerative
│  ├─ forecasting.py          # Prophet utilities
│  └─ survival.py             # lifelines utilities
├─ requirements.txt
├─ LICENSE
└─ README.md
```

> If your project is notebook-first, keep `src/` optional. The notebook includes runnable cells for each stage.

---

## Quick Start

```bash
# 1) Create environment (Python 3.10+)
python -m venv .venv
# or: conda create -n nyc-collisions python=3.10

# 2) Activate
# Windows: 
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Put dataset into ./data/
# 5) Open the notebook
jupyter lab notebooks/ML_Motor_Vehicle_Collisions_Crashes.ipynb
```

**requirements.txt (suggested)**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
prophet
lifelines
```

---

## Reproducibility

* **Random seeds** are set in modeling cells for train/test splits and model inits where supported.
* **Scaling/encoding** pipelines are fit only on training data.
* **Environment**: see `requirements.txt`. Pin versions for long-term reproducibility if needed.

---

## EDA Highlights

* **Temporal patterns:** Weekends and rush hours (~08:00–20:00) exhibit more crashes.
* **Spatial patterns:** Queens and Brooklyn contribute a large share of incidents.
* **Contributing factors:** Driver inattention, unsafe speed, and failure to yield dominate.
* **Vehicles:** Cars/SUVs are most frequently involved.
* Visuals include year/month/hour trends, borough countplots, and correlation heatmaps.

---

## Modeling

**Problem:** Binary classification of `INJURY_FLAG` (injury vs. no injury).

**Models compared:**

* Logistic Regression
* Random Forest
* Gradient Boosting
* MLPClassifier (shallow neural network)

**Evaluation:** Train/test split (80/20). Metrics—**Accuracy**, **Precision**, **Recall**, **F1**.

**Feature engineering (examples):**

* Datetime: `YEAR`, `MONTH`, `DAY_OF_WEEK`, `HOUR`
* Encoded categories: e.g., `BOROUGH` and contributing factors
* Optional scaling for linear/NN models

---

## Unsupervised, Forecasting & Survival

* **Clustering:** KMeans (plus DBSCAN/Agglomerative as experiments) on scaled numeric subsets to reveal crash pattern groups.
* **Forecasting:** Prophet for short-term daily trends with weekly seasonality.
* **Survival analysis:** Kaplan–Meier curves to estimate time between crashes (e.g., by borough).

---

## Results

**Classification (test set, indicative):**

| Model               | Accuracy | Precision |   Recall |       F1 |
| ------------------- | -------: | --------: | -------: | -------: |
| Logistic Regression |     0.83 |      0.81 |     0.82 |     0.81 |
| Random Forest       | **0.89** |  **0.88** | **0.89** | **0.88** |
| Gradient Boosting   |     0.87 |      0.86 |     0.86 |     0.86 |
| MLPClassifier       |     0.88 |      0.87 |     0.88 |     0.87 |

* **Best balance:** Random Forest (notably strong recall for injury cases).
* **Clustering:** ~4 distinct crash-behavior groups observed.
* **Forecasting:** Increasing trend with strong weekly periodicity.
* **Survival:** Shorter crash intervals in some boroughs (e.g., Brooklyn), indicating higher risk.

> Exact numbers may vary by data slice, filters, and preprocessing choices. See the notebook for the run that produced your report.

---

## Notes & Limitations

* **Missing data & inconsistency:** Cleaned/filtered where appropriate; imputation choices affect results.
* **Imbalanced targets:** `INJURY_FLAG` definition reduces extreme imbalance but still monitor class metrics.
* **External drivers not modeled:** Weather, road works, traffic volume, and holidays could improve forecasts and predictions.

---

## Roadmap

* Integrate **weather/traffic** covariates.
* Publish an **interactive dashboard** (Streamlit/Plotly).
* Explore **spatial models** (e.g., H3 grids, GWR) and **deep learning** baselines.
* Hyperparameter tuning with cross-validation for all models.

---

## How to Cite

If you use this repository, please cite:

> Warulkar, A. (2025). *Motor Vehicle Collisions Analysis using Data Science and Machine Learning*.
> Repository: [https://github.com/](https://github.com/)<your-username>/<your-repo>

---

## License

Choose a license (e.g., MIT) and add it as `LICENSE`.

---

## Acknowledgments

* NYC Open Data — *Motor Vehicle Collisions – Crashes*
* scikit-learn, Prophet, lifelines, pandas, numpy, matplotlib, seaborn
* Faculty guidance and reviewers

---

### Screenshots (Optional)

Add key plots here to make the README visual:

```
reports/figs/
  ├─ crashes_by_borough.png
  ├─ crashes_by_hour.png
  ├─ rf_feature_importance.png
  ├─ prophet_forecast.png
  └─ km_survival_curve.png
```
