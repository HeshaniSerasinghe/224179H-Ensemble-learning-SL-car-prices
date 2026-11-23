# Ensemble Learning on Sri Lankan Used-Car Prices

**Name:** Serasinghe S T H C H  **Index:** 224179H
**Module:** CM3720 Machine Learning (L4S1), Hands-On Ensemble Learning Project

I compare bagging, boosting and stacking on real Sri Lankan used-car adverts, plus a hybrid model, and show the results in a Streamlit dashboard.

**GitHub repository:** https://github.com/HeshaniSerasinghe/224179H-Ensemble-learning-SL-car-prices

**Streamlit app:** https://224179h-ensemble-learning-sl-car-prices-xsv9yhafwwkd4dug5uxkzw.streamlit.app/

| Item | Link |
|---|---|
| GitHub repository | https://github.com/HeshaniSerasinghe/224179H-Ensemble-learning-SL-car-prices |
| Live dashboard | https://224179h-ensemble-learning-sl-car-prices-xsv9yhafwwkd4dug5uxkzw.streamlit.app/ |
| Demo video (3 min) | _paste the video link here_ |

## 1. Problem and dataset
* **Task:** predict the asking price of a used car in Sri Lanka (tabular **regression**).
* **Data:** [SriLankan Vehicle (Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset)
  (Kaggle, MIT licence): 9,788 adverts scraped from **ikman.lk** and **riyasewana.com**, Dec 2024 to Feb 2025.
  Price is in **LKR lakhs** (1 lakh = Rs. 100,000).
* **Data quality findings** (details in the notebook):
  * `Mileage` equals exactly `11,000 x age` in every row, so it is not real information. I dropped it.
  * 252 re-posted adverts were removed so copies cannot leak from train to test.
  * A few inconsistent engine capacities (e.g. a luxury hybrid listed as 600 cc) were kept as noise.
* **Target:** modelled as `log(1 + price)` (right-skewed, relative errors matter).

## 2. What I built
| Family | Models |
|---|---|
| Baselines | Ridge regression, single Decision Tree |
| **Bagging** | Bagged decision trees, **Random Forest** (OOB error, number of trees B, tree correlation rho experiments) |
| **Boosting** | **XGBoost**, **LightGBM**, **CatBoost** (one-hot and native categoricals) |
| **Stacking** | RF + XGBoost + CatBoost + SVR + ANN with a Ridge meta-learner |
| **Hybrid (RF + ANN)** | Blend, residual correction, and RF-feature ANN; the design is chosen by cross-validation |

All models share one pipeline (feature engineering, encoding, scaling only where needed), the same stratified
80/20 split, `GridSearchCV` tuning (3-fold, RMSLE), and are evaluated with RMSLE, MAE, RMSE, R2 and MAPE.
Class imbalance / SMOTE is not applicable (regression).

## 3. Main results (held-out test set, 1,908 cars)
| Rank | Model | RMSLE | MAE (lakhs) | RMSE (lakhs) | R2 | MAPE |
|---|---|---|---|---|---|---|
| 1 | Stacking (RF+XGB+Cat+SVR+ANN) | 0.198 | 5.39 | 14.00 | 0.891 | 12.3% |
| 2 | XGBoost | 0.200 | 5.64 | 14.06 | 0.890 | 12.7% |
| 4 | Hybrid RF+ANN (blend, CV-selected) | 0.203 | 5.39 | 14.05 | 0.890 | 12.6% |
| 5 | CatBoost | 0.205 | 6.00 | 14.45 | 0.884 | 13.5% |
| 10 | LightGBM | 0.211 | 5.88 | 14.38 | 0.885 | 13.7% |
| 11 | Random Forest | 0.212 | 5.40 | 14.71 | 0.879 | 12.8% |
| 12 | ANN (MLP) | 0.214 | 6.24 | 14.43 | 0.884 | 14.1% |
| 14 | Ridge (linear baseline) | 0.252 | 7.26 | 17.12 | 0.837 | 17.0% |
| 15 | Decision Tree | 0.277 | 7.64 | 19.61 | 0.786 | 18.1% |

(The full 15-model table is in `results/model_comparison.csv` and in the notebook.)

**Main points**
* Every ensemble clearly beats the single tree and the linear model (paired bootstrap, 95% interval excludes 0).
* XGBoost, stacking and the hybrid are significantly better than a Random Forest, but the ranking *among* the best few is statistically a tie.
* All models stop at about 12% average error. That is the irreducible error: the data has no real mileage, trim or condition.
* Year of manufacture is the strongest driver, followed by brand, gearbox and engine size.
* Trees cannot extrapolate (they flatten for the newest cars) while the ANN keeps the trend, which is useful in a hybrid.

## 4. Repository structure
```
.
|-- Ensemble_Learning_SL_Car_Prices.ipynb   # full pipeline, experiments, written analysis
|-- ensemble_utils.py                       # feature engineering + hybrid model class (needed by the .pkl files)
|-- app.py                                  # Streamlit dashboard
|-- requirements.txt
|-- data/car_price_dataset.csv              # Kaggle dataset
|-- models/*.pkl, metadata.json             # saved pipelines used by the dashboard
`-- results/*.csv                           # metrics, importances, experiment outputs read by the dashboard
```

## 5. Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py                         # dashboard
jupyter notebook Ensemble_Learning_SL_Car_Prices.ipynb   # re-run the experiments (about 30 minutes)
```
Python 3.12 was used. The saved models need the pinned `scikit-learn`, `xgboost`, `lightgbm` and `catboost` versions in `requirements.txt`.

## 6. Dashboard
* **Overview**, **Model comparison** (metric picker, error by price segment, predicted vs actual),
  **Feature importance** (permutation and impurity), **Bagging lab** (OOB vs number of trees, tree correlation),
  **Live prediction** (enter a car, get a price from several models, the model spread and the Random Forest tree spread).

The live dashboard is already deployed on Streamlit Community Cloud with Python 3.12. The link is in the table at the top.

## 7. AI usage acknowledgment
I used an AI assistant (Claude, in the Cursor editor) to clarify concepts (bias-variance, OOB error, stacking, the hybrid designs) and for coding help, and the Streamlit dashboard (`app.py`), and the README. I ran the experiments, checked the outputs, and wrote the interpretation from those results.

## 8. References
1. Dataset, Nirmal, P. SriLankan Vehicle (Car) Price Dataset, Kaggle (scraped from ikman.lk and riyasewana.com), MIT licence.
2. Lecture and Lab notes, CM3720 Machine Learning (ensemble learning, bagging and random forests).

