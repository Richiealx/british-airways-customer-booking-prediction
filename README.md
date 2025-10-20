# ✈️ British Airways – Customer Booking Propensity Prediction  

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange.svg)  
![Forage](https://img.shields.io/badge/Forage-Simulation-green.svg)  
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-yellow.svg)  

---

## Project Overview  

This repository presents a **machine learning solution** developed as part of the **British Airways Forage Job Simulation**.  
The project predicts the **likelihood of a customer completing a flight booking** based on behavioural, temporal, and ancillary features.  

The analysis demonstrates **end-to-end applied data science**:  
- Exploratory Data Analysis (EDA)  
- Feature engineering (customer behaviour, route demand, ancillary purchases)  
- Reproducible scikit-learn pipelines  
- Balanced model training (Random Forest)  
- Business-ready metrics (ROC-AUC, PR-AUC, Precision@k, Lift)  
- Actionable commercial recommendations  

> **Headline results:**  
> - Holdout ROC-AUC: **0.794**  
> - Holdout PR-AUC: **0.392**  
> - Precision@Top-10%: **43.8%** (~**2.9×** lift vs baseline booking rate of 15%)  

---

## 1) Business Problem  

Converting flight searchers into confirmed bookings is a **core revenue lever**.  
The goal is to **rank customers by booking likelihood**, enabling British Airways to focus low-cost marketing outreach (email, app notifications, loyalty nudges) on the **highest-propensity segments**.  

- **Target variable:** `booking_complete` (1 = booked, 0 = not booked)  
- **Baseline booking rate:** ~15%  

---

## 2) Data  

Dataset: **50,000 simulated booking records** provided by Forage.  

**Key raw features**  
- Customer: `num_passengers`, `booking_origin`  
- Booking behaviour: `purchase_lead`, `length_of_stay`, `trip_type`, `sales_channel`  
- Flight context: `route`, `flight_hour`, `flight_day`, `flight_duration`  
- Ancillaries: `wants_extra_baggage`, `wants_preferred_seat`, `wants_in_flight_meals`  
- Target: `booking_complete`  

**Note**: The dataset is not redistributed here. Please download `customer_booking.csv` from Forage and place it under:  




---

##  3) Methodology  

### Feature Engineering  
- **Route parsing** → `origin`, `destination`  
- **Behavioural bins** → `lead_bin`, `stay_bin`, `duration_bin`  
- **Temporal buckets** → `flight_daypart` (overnight/morning/afternoon/evening)  
- **Interaction** → `trip_x_channel` (trip type × sales channel)  
- **Commercial intent** → `ancillaries_count` (sum of extra services)  
- **Demand proxy** → `route_popularity` (frequency encoding of routes)  

### Preprocessing Pipeline  
- Numeric → median imputation + scaling  
- Categorical → cast to strings + `"missing"` token + One-Hot Encoding  
- Wrapped in a single **scikit-learn `Pipeline`** for reproducibility  

### Model  
- `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)`  

### Validation Strategy  
- **80/20 stratified train/holdout split**  
- **3-fold Stratified Cross-Validation** on training set  
- Evaluation focused on **PR-AUC and Precision@k** due to class imbalance  

---

## 4) Results  

### Cross-Validation (train set)  
- ROC-AUC: **0.7765 ± 0.0017**  
- PR-AUC: **0.3668 ± 0.0049**  

### Holdout (test set)  
- ROC-AUC: **0.7942**  
- PR-AUC: **0.3918**  

### Targeting Efficiency (holdout)  

| Segment   | Contacts | Precision@k | Lift vs 15% baseline |
|-----------|---------:|-------------:|---------------------:|
| Top **5%**   | 500   | **0.4920** | **3.29×** |
| Top **10%**  | 1,000 | **0.4380** | **2.93×** |
| Top **20%**  | 2,000 | **0.3795** | **2.54×** |

**Operating point (Top-10%)**  
- Threshold ≈ 0.36  
- Precision = 0.438  
- Recall = 0.294  
- Confusion matrix: [[TN=7940, FP=564], [FN=1056, TP=440]]  

### Key Drivers (top feature importances)  
- `purchase_lead`  
- `length_of_stay`  
- `flight_hour`  
- `booking_origin`  
- `route_popularity`  
- `flight_day`  
- `num_passengers`  
- `flight_duration`  
- `ancillaries_count`  

---

##  5) Visualisations  

- Precision–Recall Curve → `figures/pr_curve.png`  
- ROC Curve → `figures/roc_curve.png`  
- Feature Importances → `figures/feature_importances_top15.png`  

<p align="center">
  <img src="figures/feature_importances_top15.png" width="70%">
</p>

---

##  6) Business Recommendations  

- **Activate Top-Decile**: Target the top 10% by propensity score with **low-cost channels** (email/app/loyalty). Expect ~3× uplift in booking conversion.  
- **Offer Design**: A/B test bundles (extra baggage, seat selection) among high-propensity customers; optimise by market/route.  
- **MLOps & Monitoring**:  
  - Retrain every 3-6 months on fresh data  
  - Monitor PR-AUC and Precision@k for drift  
  - Add fairness checks across regions and customer groups  

---

## 7) Reproducibility  

### Local Setup  
```bash
git clone https://github.com/Richiealx/british-airways-booking-propensity-ml.git
cd ba-booking-propensity-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


Repository Structure
notebooks/British_Airways-Task 2_Predicting_customer_buying_behaviour.ipynb   # Main analysis
figures/                                               # PR, ROC, feature importances
report/Predicting-Customer-Booking-Behavior.pptx                     # PowerPoint Presentation of the Project
src/ (optional)                                        # Helper functions if modularised
data/ (gitignored)                                     # Place customer_booking.csv here

---

Disclaimer

This repository is for educational and portfolio purposes only.
It uses a simulation dataset from Forage and does not represent British Airways’ proprietary systems, data, or strategy.

