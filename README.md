#  Asteroid Threat Predictor

This project predicts whether a Near-Earth Object (NEO) is potentially hazardous to Earth based on its physical and orbital characteristics, using Machine Learning. Inspired by real-world planetary defense challenges, it leverages NASA-like data to simulate threat classification for asteroids.

---

##  Project Motivation

Accurately identifying hazardous asteroids is crucial for planetary safety. Most asteroids are harmless, but the few that come close to Earth can pose significant risks. This project simulates how data-driven techniques can improve threat detection for early warning systems.

---

##  Project Structure

Asteroid-Threat-Predictor/
README.md
requirements.txt
.gitignore

data/
    Group5.csv

notebooks/
    Asteroid_Threat_Predictor.ipynb

models/
    asteroid_rf_model.pkl
    transformer.pkl

scripts/
    asteroid_threat_predictor.py

images/
    corr_heatmap.png
    class_balance.png
    feature_importance.png

---

##  Dataset Overview

- **Rows:** ~338,000
- **Columns:**
    - `absolute_magnitude` — intrinsic brightness of the asteroid
    - `estimated_diameter_min` / `max` — size estimates in kilometers
    - `relative_velocity` — speed relative to Earth (km/s)
    - `miss_distance` — closest distance to Earth (km)
    - `is_hazardous` — target label (True/False)

Class balance (before preprocessing):

| Class            | Count    | Percentage |
|------------------|----------|------------|
| Not Hazardous    | 294,000  | 87%        |
| Hazardous        | 44,000   | 13%        |

![Class Balance](images/class_balance1.png)
![Class Balance](images/class_balance2.png)
---

##  Exploratory Data Analysis (EDA)

### Correlation Heatmap

- Velocity and miss distance have interesting relationships.
- Brighter asteroids (lower magnitude) tend to be slightly larger and faster.

![Correlation Heatmap](images/Before_Corr_Heatmap.png)

---

### Feature Distributions

- **Absolute Magnitude**: hazardous asteroids tend to be brighter.
- **Estimated Diameter**: hazardous asteroids are typically larger.
- **Velocity Distance Ratio**: hazardous asteroids often travel faster per distance.

---

##  Feature Engineering

Added new features:

- **estimated_diameter_avg** = average diameter
- **diameter_range** = difference between min and max diameters
- **velocity_distance_ratio** = velocity / miss distance
- **log_miss_distance** and **log_relative_velocity** for handling skewness

These engineered features helped boost model performance and reduce skewed distributions.

![Correlation Heatmap](images/After_Corr_Heatmap.png)

---

##  Data Processing Pipeline

✔ Missing values handled via median imputation  
✔ Class imbalance addressed with SMOTE  
✔ Features scaled using QuantileTransformer for better handling of outliers  
✔ Feature selection via SelectKBest reduced the model to the **6 most predictive features**:

- absolute_magnitude
- estimated_diameter_avg
- diameter_range
- velocity_distance_ratio
- log_miss_distance
- log_relative_velocity

---

##  Model Selection

Tested several models:

- Logistic Regression
- Decision Tree
- Naive Bayes
- Random Forest  *(best performer)*
- XGBoost *(More Accuracy but requires high device specifications)*

---

##  Best Model Results

Random Forest achieved:

| Metric      | Score |
|-------------|-------|
| Accuracy    | 0.757 |
| Precision   | 0.340 |
| Recall      | 0.960 |
| F1-score    | 0.502 |

*Note:* The model prioritizes high recall for hazardous asteroids — essential in a real-world scenario where **missing a threat is more dangerous than false alarms.**

---

##  Feature Importance

![Feature Importance](images/feature_importance.png)

- Absolute magnitude and estimated diameter avg were the most significant predictors.

---

##  Saving Models

The pipeline saves:

- Trained Random Forest model → `models/asteroid_rf_model.pkl`
- Fitted transformer for scaling → `models/transformer.pkl`

These artifacts allow seamless predictions without retraining.

---

##  How to Run the Pipeline

1. Clone this repo:
``` bash
    git clone https://github.com/yourusername/Asteroid-Threat-Predictor.git
```

2. Install dependencies:
    pip install -r requirements.txt

3. Run the ML pipeline script:
    cd scripts
    python asteroid_threat_predictor.py

##  Future Work
     Integrate NASA NeoWs live API for real-time asteroid predictions
     Build a web UI using Streamlit for easy input
     Explore ensemble methods for higher precision
     Deploy as a REST API
     Investigate deep learning architectures for more complex relationships
     
## Author
Sarib Khan
