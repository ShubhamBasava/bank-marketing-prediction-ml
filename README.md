# Portuguese Bank Term Deposit Prediction (PRCP-1000)

## Project Overview

Banks frequently run marketing campaigns to promote term deposit products. Contacting every customer is expensive and inefficient. This project builds a machine learning model to predict whether a customer is likely to subscribe to a term deposit, enabling the bank to focus outreach efforts on high-probability customers.

**Goal:**  
Predict customer subscription to a term deposit (`y` = yes/no)

**Problem Type:**  
Binary Classification

**Business Impact:**
- Reduce unnecessary marketing calls
- Increase campaign conversion rates
- Optimize resource allocation and marketing spend

---

## Dataset

- **Source**: Portuguese Bank Marketing Dataset (UCI-style / PRCP-1000)
- **Files**: `bank-full.csv`, `bank-additional-full.csv`, and related documentation
- **Target Variable**: `y` (yes / no)
- **Key Features**:
  - Demographic: `age`, `job`, `marital`, `education`
  - Financial: `default`, `housing`, `loan`
  - Campaign: `contact`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`, `poutcome`
  - Economic indicators: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

**Important Note:**  
The feature `duration` (last contact duration) was **removed** before modeling to prevent data leakage, as this information is only available after a call has been completed.

---

## Libraries & Tools Used

- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Preprocessing**: LabelEncoder, StandardScaler
- **Modeling**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Evaluation**: Accuracy, Classification Report, Confusion Matrix, ROC-AUC
- **Model Persistence**: joblib

---

## Project Pipeline

1. **Problem Understanding** – Define business objective and success metrics  
2. **Data Loading & Exploration** – Load dataset, inspect structure and distributions  
3. **Exploratory Data Analysis (EDA)** – Analyze target balance, feature relationships, and missing values  
4. **Data Preprocessing**
   - Treat missing categorical values as `"unknown"`
   - Encode categorical variables (Label Encoding)
   - Remove `duration` to avoid leakage
   - Scale numerical features where required
5. **Train-Test Split** – Stratified split to preserve class distribution  
6. **Model Building** – Train multiple classification algorithms  
7. **Model Evaluation** – Primary metric: **ROC-AUC** (due to class imbalance)  
8. **Hyperparameter Tuning** – Optimize top models using GridSearchCV / cross-validation  
9. **Final Model Selection** – Choose best model based on ROC-AUC  
10. **Model Persistence** – Save final model as `final_bank_model.pkl`  
11. **Business Insights & Recommendations**

---

## Models Evaluated

| Model                  | Type                  | Notes                                      |
|------------------------|-----------------------|--------------------------------------------|
| Logistic Regression    | Linear                | Interpretable baseline                     |
| Decision Tree          | Tree-based            | Easy to interpret                          |
| Random Forest          | Ensemble (Bagging)    | Strong performance                         |
| **Gradient Boosting**  | Ensemble (Boosting)   | **Selected as final model**                |

Additional models were evaluated during the process; ensemble methods consistently outperformed linear models.

### Final Model Performance

**Best Model:** Gradient Boosting  
**Final ROC-AUC:** **0.9476**

---

## Key Challenges & Solutions

| Challenge                                                                 | Solution Applied                                                                 |
|---------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Missing values in categorical features (`job`, `education`, `default`, etc.) | Treated as separate category `"unknown"` to preserve information                 |
| High missing values in `default`                                          | Retained `"unknown"` category instead of dropping rows                           |
| Class imbalance in target (`y`)                                           | Used **ROC-AUC** as primary evaluation metric                                    |
| Data leakage risk from `duration`                                         | Completely removed `duration` before training                                    |
| Categorical variable handling                                             | Applied Label Encoding for model compatibility                                   |
| Potential overfitting                                                     | Used cross-validation and hyperparameter tuning                                  |
| Model selection complexity                                                | Evaluated multiple models with consistent metrics and selected by ROC-AUC        |
| Hyperparameter sensitivity                                                | Applied GridSearchCV to optimize performance                                     |
| Feature importance understanding                                          | Leveraged tree-based feature importance                                          |

---

## Key Insights

- Ensemble models (Random Forest and Gradient Boosting) significantly outperformed linear models due to their ability to capture non-linear relationships.
- Removing the `duration` feature was critical to ensure realistic, deployable performance.
- Hyperparameter tuning improved both performance and model stability.
- Economic indicators and previous campaign outcomes are strong predictors of subscription likelihood.

---

## Business Impact & Recommendations

- The model can accurately identify customers with a high probability of subscribing to term deposits.
- Enables **targeted marketing campaigns**, reducing the volume of unnecessary calls.
- Improves conversion rates and optimizes marketing resource allocation.
- Supports data-driven decision making in campaign planning.

**Final Recommendation:**
- Deploy the Gradient Boosting model into the bank’s marketing pipeline.
- Continuously retrain the model with new campaign data to maintain performance over time.

---

## How to Reproduce

### Requirements
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

### Steps
1. Open the notebook `PRCP_1000_Portuges_Bank.ipynb`
2. Download and unzip the dataset
3. Run data loading, EDA, and preprocessing cells
4. Train and evaluate multiple models
5. Perform hyperparameter tuning
6. The final model is saved as:
   ```python
   import joblib
   joblib.dump(final_model, "final_bank_model.pkl")
   ```

### Loading the Saved Model
```python
import joblib

model = joblib.load("final_bank_model.pkl")
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

---

## Project Structure

```
├── PRCP_1000_Portuges_Bank.ipynb          # Main analysis notebook
├── final_bank_model.pkl                   # Trained Gradient Boosting model
├── Data/
│   ├── bank-full.csv
│   ├── bank-additional/
│   │   └── bank-additional-full.csv
│   └── bank-names.txt / documentation
└── README.md              # This file
```

---

## Conclusion

This project successfully developed a high-performing machine learning solution for predicting term deposit subscriptions. After rigorous evaluation and hyperparameter tuning, **Gradient Boosting** achieved a strong **ROC-AUC of 0.9476**.

By carefully handling data leakage, class imbalance, and categorical features, the model delivers realistic performance suitable for real-world deployment. The solution enables the bank to run more efficient, targeted marketing campaigns, reduce operational costs, and improve conversion rates.

---

**Project Code**: PRCP-1000  
**Domain**: Banking / Marketing  
**Task Type**: Binary Classification (Term Deposit Subscription)  
**Final Model**: Gradient Boosting  
**Key Metric**: ROC-AUC = 0.9476  
**Framework**: Scikit-learn
