# Formula 1 Pit Stop Analysis with Explainable AI (XAI)

This project analyzes a dataset of Formula 1 pit stop times from 1950 to 2024. The goal is to build a machine learning model that can predict the average pit stop time and then use methods from Explainable AI (XAI) to understand which factors have the greatest impact.



---

### Table of Contents
1.  [Prerequisites](#prerequisites)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
4.  [Results](#results)
5.  [Conclusion](#conclusion)

---

### Prerequisites
To run the Jupyter notebook and reproduce the analysis, the following Python libraries must be installed:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`

You can install them using `pip`:

` ``bash
pip install pandas numpy matplotlib scikit-learn scipy
` ``

---
### Dataset
The dataset, `Formula1_Pitstop_Data_1950-2024_all_rounds.csv`, contains historical data on Formula 1 pit stops. Before the analysis, rows with missing values in `AvgPitStopTime` were removed.

The following features were used to predict the average pit stop time (`AvgPitStopTime`):
* **`Season`**: The season year of the race.
* **`Laps`**: The total number of laps in the race.
* **`Round`**: The race's round in the season.
* **`TotalPitStops`**: The total number of pit stops.
* **`Position`**: The driver's final position.

---
### Methodology
The analysis follows a structured process for machine learning and model explanation.

1.  **Data Preparation and Modeling**:
    * The dataset was split into training, validation, and test sets.
    * Three different regression models were trained to predict `AvgPitStopTime`:
        * Linear Regression.
        * Random Forest Regressor (with hyperparameter tuning via `GridSearchCV`).
        * Support Vector Regressor (SVR) (with hyperparameter tuning via `GridSearchCV`).

2.  **Explainable AI (XAI)**:
    * To understand the complex Random Forest model ("black box"), several global, model-agnostic methods were used:
        * **Partial Dependence Plots (PDP)**: Visualizes the average effect a feature has on the prediction.
        * **Accumulated Local Effects (ALE)**: An improved version of PDP that is more robust against correlated features.
        * **Permutation Feature Importance (PFI)**: Ranks feature importance by measuring how much the model's performance decreases when a feature is randomly shuffled.
        * **Surrogate Model**: A simpler decision tree was trained to mimic the predictions of the Random Forest model and provide a more interpretable approach.

---
### Results
* **Model Performance**: **Random Forest** was by far the best model, with an R² score of **0.772** on the test set. Linear Regression and SVR performed very poorly, indicating a non-linear relationship in the data.

* **Feature Importance (PFI)**: Permutation importance showed that **`Season`** (season year) and **`Laps`** (number of laps) were the most important features for predicting pit stop time.

* **Feature Effects (PDP/ALE)**: Both PDP and ALE plots showed a clear **positive trend** for `Season`, meaning the average pit stop time has increased over the years. For `TotalPitStops`, the relationship was more complex and non-linear.

---
### Conclusion
The analysis shows that a Random Forest model is well-suited for predicting the average pit stop time in Formula 1. Using XAI methods, we have uncovered that the **season year** and the **number of laps** in the race are the most crucial factors. The increasing trend in pit stop time across seasons is an interesting finding that may be due to changes in regulations, technology, or strategy in the sport that are not directly captured by the other variables in the dataset.

