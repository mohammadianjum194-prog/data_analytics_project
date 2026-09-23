# 🏥 CareSight AI
### *Healthcare Intelligence, Simplified.*

> **IBM SkillsBuild — Data Analytics with AI · Internship Capstone Project · 2026**

---

## 📌 Overview

**CareSight AI** is an end-to-end interactive healthcare analytics platform built with **Streamlit** and Python. It combines exploratory data analysis, dynamic Plotly visualisations, and machine learning to help healthcare analysts understand patient admission patterns and predict hospital readmissions — all within a single, browser-based dashboard.

| | |
|---|---|
| **App file** | `Anjum_CareSightAI.py` |
| **Dataset** | `Hospital Patient .csv` or 'https://www.kaggle.com/datasets/sathwiknomula/hospital-patient-records-dataset' |
| **Report** | `Anjum_CareSightAI_ProjectReport.docx` |
| **Python** | 3.12.x |
| **Framework** | Streamlit · Plotly · scikit-learn |

---

## 🚑 Problem Statement

Hospital readmissions are a significant indicator of patient outcome quality and healthcare costs. Identifying patients at risk of readmission before discharge allows clinicians to intervene early. This project addresses the challenge of **predicting hospital readmission risk** using patient demographic, clinical, and financial data, while also providing an analytics layer for hospital performance monitoring.

---

## 🎯 Objectives

- Build an interactive dashboard to explore patient admission patterns across conditions, states, and years
- Enable drill-down analytics via multi-dimensional filters
- Train and evaluate ML models (Logistic Regression & Random Forest) for readmission prediction
- Present model performance transparently with confusion matrices, ROC curves, and feature importance
- Package everything as a deployable, single-file Streamlit application

---

## 📂 Project Structure

```
project/
├── Anjum_CareSightAI.py              ← Main Streamlit application
├── Hospital Patient .csv             ← Dataset (984 patient records)
├── requirements.txt                  ← Python package dependencies
├── Anjum_CareSightAI_ProjectReport.docx  ← Full project report
├── README.md                         ← This file
└── screenshots/                      ← UI screenshots (see below)
    ├── dashboard.png
    ├── patient_analytics.png
    ├── ai_insights.png
    └── model_performance.png
```

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| **File** | `Hospital Patient .csv` |
| **Records** | 984 patients |
| **Columns** | 15 |
| **Target** | `Readmission` (Yes / No) |
| **Year Range** | 2022 – 2025 |
| **Medical Conditions** | 14 unique |
| **States Covered** | 28 Indian states |

<details>
<summary>Column descriptions</summary>

| Column | Type | Description |
|--------|------|-------------|
| `Patient_ID` | Integer | Unique identifier — excluded from ML |
| `Age` | Integer | Patient age in years |
| `Gender` | String | Male / Female |
| `Condition` | String | Diagnosed medical condition |
| `Medication` | String | Treatment administered — excluded from ML |
| `Admission_Date` | Date | DD-MM-YYYY |
| `Discharge_Date` | Date | DD-MM-YYYY |
| `Patient_State` | String | Indian state of residence |
| `Year_of_Admission` | Integer | Year of admission |
| `Length_of_Stay` | Integer | Days in hospital |
| `Readmission` | String | **Target** — Yes / No |
| `Outcome` | String | Recovered / Stable |
| `Satisfaction` | Integer | Score 1–5 |
| `Insurance_Claimed` | String | Yes / No |
| `Total_Cost` | Integer | Total cost in ₹ |

</details>

---





---

## ✨ Features

| Page | Highlights |
|------|-----------|
| **📊 Dashboard** | 5 KPI cards · 7 interactive Plotly charts · yearly trend line |
| **🔍 Patient Analytics** | 7 filters · dynamic KPI refresh · 8+ drill-down charts |
| **🤖 AI Insights** | Dual-model live prediction · probability comparison chart |
| **📈 Model Performance** | Metrics table · confusion matrices · ROC curves · feature importance |
| **ℹ️ About Project** | Dataset summary · tech stack · ML workflow · limitations · disclaimer |

---

## 🔬 ML Workflow

```
Hospital Patient .csv  (984 records)
        │
        ▼
  Data Cleaning
  ├─ Strip whitespace from column names & string values
  ├─ Parse Admission / Discharge dates → derive Length_of_Stay
  └─ Binary-encode target  (Readmission: Yes → 1, No → 0)
        │
        ▼
  Feature Engineering
  ├─ Age-group binning
  ├─ Label-encode all categorical features
  └─ Drop Patient_ID & Medication  (leakage prevention)
        │
        ▼
  Train / Test Split  —  80 % train · 20 % test · stratified
        │
        ▼
  Model Training
  ├─ Logistic Regression  (max_iter = 1000)
  └─ Random Forest         (n_estimators = 100)
        │
        ▼
  Evaluation
  ├─ Accuracy · Precision · Recall · F1 · ROC-AUC
  ├─ Confusion Matrix  (Plotly heatmap)
  ├─ Classification Report
  ├─ Feature Importance  (Random Forest)
  └─ ROC Curves  (with AUC)
        │
        ▼
  Streamlit App  →  Dashboard · Analytics · Prediction · Performance
```

**Features used in models:**

`Age` · `Gender` · `Condition` · `Patient_State` · `Length_of_Stay` · `Outcome` · `Satisfaction` · `Insurance_Claimed` · `Total_Cost` · `Year_of_Admission`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12.x |
| **Web Framework** | Streamlit ≥ 1.32 |
| **Data** | Pandas ≥ 2.0 · NumPy ≥ 1.24 |
| **Visualisation** | Plotly ≥ 5.18 |
| **ML** | scikit-learn ≥ 1.3 |

---

## ⚙️ Installation & Run

### Prerequisites
- Python **3.12.x**
- pip (bundled with Python)

### 1 — Get the files

Download or clone all project files into a single folder.

### 2 — Verify dataset placement

The CSV must be in the **same folder** as the app script, with its exact name (note the space before `.csv`):

```
your-folder/
├── Anjum_CareSightAI.py
├── Hospital Patient .csv   ← exact name required
├── requirements.txt
└── README.md
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Launch the app

```bash
streamlit run Anjum_CareSightAI.py
```

The app opens automatically at **`http://localhost:8501`**.

---

## ⚠️ Limitations

- Dataset is educational/synthetic — not sourced from a live hospital EHR system
- Models have **not** been clinically validated
- No real-time data pipeline or database connectivity
- Basic feature engineering; no advanced imputation, scaling, or class-imbalance handling (SMOTE / `class_weight`)
- Prediction reliability is constrained by the small sample size (984 records)

---

## 🚀 Future Scope

- Gradient-boosted models — XGBoost, LightGBM, Neural Networks
- SHAP explainability for model transparency
- Real hospital EHR / FHIR API integration
- Automated PDF report export from the dashboard
- Geospatial heatmap of admissions by Indian state
- CI/CD deployment on Streamlit Cloud or IBM Cloud

---

## ⚕️ Healthcare Disclaimer

> **This project is for educational and analytical purposes only.**
> Predictions are based on historical sample data and do **not** constitute medical diagnosis or clinical advice.
> Always consult a qualified healthcare professional for any medical decisions.

---

## 📄 Report & Documentation

Full methodology, data analysis, model evaluation, and results are documented in:

📄 **`Anjum_CareSightAI_ProjectReport.docx`**

---

*Made by **Anjum** · IBM SkillsBuild Data Analytics with AI · 2026*
