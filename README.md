# Credit Risk Analysis and Modeling

This repository contains a comprehensive credit risk analysis and machine learning modeling project that predicts loan defaults using various algorithms and provides business insights for loan approval strategies.

## Project Overview

The project analyzes credit risk data to build predictive models that can help financial institutions make informed loan approval decisions. It includes data preprocessing, feature engineering, multiple machine learning models, model explainability, and business strategy optimization with expected loss calculations.

## Dataset

The dataset (`credit_risk_dataset.csv`) contains loan application data with the following features:

| Feature Name | Description |
|--------------|-------------|
| person_age | Age of the applicant |
| person_income | Annual income |
| person_home_ownership | Home ownership status |
| person_emp_length | Employment length (in years) |
| loan_intent | Purpose of the loan |
| loan_grade | Loan grade (A-G) |
| loan_amnt | Loan amount |
| loan_int_rate | Interest rate |
| loan_status | Target variable (0: no default, 1: default) |
| loan_percent_income | Loan amount as percentage of income |
| cb_person_default_on_file | Historical default record |
| cb_preson_cred_hist_length | Credit history length |

## Project Structure

```
Credit-Risk/
├── notebook.ipynb              # Main analysis notebook
├── EDA.ipynb                   # Exploratory data analysis
├── functions.py                # Utility functions
├── README.md                   # Project documentation
├── data_models/                # Saved models and data
│   ├── best_nn_model.keras     # Best neural network model
│   ├── best_tree_model.pkl     # Best tree-based model (LightGBM)
│   ├── xgb_model.json          # XGBoost model for imputation
│   ├── credit_risk_dataset.csv # Raw dataset
│   └── final_data.csv          # Processed dataset
└── __pycache__/                # Python cache files
```

## Analysis Pipeline

### 1. Data Preprocessing
- **Outlier Removal**: Removes extreme values for age (>85), income (>200k), employment length (>18), and loan amount (>28k)
- **One-Hot Encoding**: Categorical variables (home ownership, loan intent)
- **Ordinal Encoding**: Loan grades (A=1 through G=7)
- **Binary Encoding**: Historical default flag (Y=1, N=0)

### 2. Missing Value Imputation
- **Employment Length**: Mean imputation (~4.6 years)
- **Interest Rate**: XGBoost regression model prediction (MAE ~0.78%)

### 3. Model Development

#### Tree-Based Models
- **Random Forest**: Ensemble baseline model (300 estimators, max_depth=5)
- **LightGBM**: Best performing tree model (F1-score: ~0.82)
- **XGBoost**: Alternative gradient boosting approach
- **Ensemble**: Voting classifier combining all three models

#### Neural Network
- **Architecture**: 5-layer deep network (128→64→32→16→1)
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Optimizer**: RMSprop with binary crossentropy loss
- **Regularization**: Early stopping (patience=3)
- **Scaling**: RobustScaler for feature normalization

### 4. Model Evaluation
- **Cross-Validation**: 5-fold stratified CV
- **Metrics**: F1-score, Precision, Recall, Accuracy
- **Calibration Curves**: Model reliability assessment for both LightGBM and Neural Network

### 5. Model Explainability

#### SHAP Analysis (Neural Network)
- **Global Importance**: Feature ranking across all predictions
- **Local Explanations**: Individual prediction breakdowns
- **Top Features**: Loan grade, interest rate, loan amount, income percentage
- **Visualization**: Summary plots and feature importance charts

#### LIME Analysis (Neural Network)
- **Local Interpretability**: Instance-level explanations for random samples
- **Feature Contributions**: Positive/negative impact visualization
- **Random Sample Analysis**: Detailed breakdown of 4 selected cases

### 6. Business Strategy Optimization
- **Acceptance Rate Analysis**: Testing rates from 10% to 100%
- **Risk Metrics**: Bad rate, average loan amount, average interest rate
- **Expected Loss Calculation**: PD × EAD × LGD (60% Loss Given Default)
- **Revenue Estimation**: Principal × Interest Rate
- **Net Profit Optimization**: Revenue - Expected Loss

## Key Results

### Model Performance
- **Best Model**: LightGBM classifier
- **Cross-Validation F1-Score**: ~0.82
- **Test Performance**: Well-calibrated probability predictions
- **Comparison**: LightGBM outperformed Random Forest, XGBoost, and Neural Network

### Feature Importance (SHAP Analysis)
Top predictive features for credit risk:
1. Loan grade (most important)
2. Interest rate
3. Loan amount
4. Loan percent income
5. Employment length
6. Credit history length
7. Person age
8. Annual income

### Business Strategy Insights
- **Expected Loss Analysis**: Total portfolio expected loss calculated
- **Optimal Thresholds**: Risk-based acceptance rate strategies
- **Revenue vs Risk**: Trade-off analysis between acceptance rates and profitability
- **Portfolio Metrics**: 3-graph visualization showing Expected Loss, Revenue, and Net Profit

## Expected Loss Framework

### Formula Implementation
```
Total Expected Loss = PD × EAD × LGD
```

Where:
- **PD (Probability of Default)**: Model prediction (0-1)
- **EAD (Exposure at Default)**: Loan amount
- **LGD (Loss Given Default)**: 60% (industry standard assumption)

### Business Metrics
- **Total Principal Lent**: Number of accepted loans × Average loan amount
- **Total Expected Revenue**: Principal × Average interest rate
- **Net Expected Profit**: Revenue - Expected Loss

## Usage

### Requirements
```bash
pip install pandas numpy scikit-learn lightgbm xgboost tensorflow
pip install shap lime matplotlib seaborn joblib
```

### Running the Analysis

1. **Data Loading and Preprocessing** (Cells 1-15):
   ```python
   original_data = pd.read_csv('credit_risk_dataset.csv')
   # Execute outlier removal, encoding, and missing value imputation
   ```

2. **Model Training** (Cells 16-45):
   ```python
   # Train LightGBM, XGBoost, Random Forest, and Neural Network
   # Compare performance using cross-validation
   ```

3. **Model Evaluation** (Cells 46-55):
   ```python
   # Generate calibration curves and performance metrics
   # Evaluate best model on test set
   ```

4. **Explainability Analysis** (Cells 56-75):
   ```python
   # SHAP analysis for global and local explanations
   # LIME analysis for individual predictions
   ```

5. **Business Strategy** (Cells 76-85):
   ```python
   # Strategy table generation with acceptance rates
   # Expected loss and profit calculations
   # Visualization of business metrics
   ```

### Loading Saved Models
```python
import joblib
from tensorflow import keras

# Load best tree model (LightGBM)
best_tree_model = joblib.load('data_models/best_tree_model.pkl')

# Load neural network
best_nn_model = keras.models.load_model('data_models/best_nn_model.keras')

# Load XGBoost model for imputation
xgb_imputation_model = xgb.XGBRegressor()
xgb_imputation_model.load_model('data_models/xgb_model.json')
```

## Model Interpretation

### Risk Factors (Increase Default Probability)
- **Higher loan grades** (E, F, G grades)
- **High loan-to-income ratio** (loan_percent_income)
- **History of defaults** (cb_person_default_on_file = 1)
- **Shorter employment history** (person_emp_length)
- **Higher interest rates** (loan_int_rate)

### Protective Factors (Decrease Default Probability)
- **Lower loan grades** (A, B, C grades)
- **Home ownership** (especially MORTGAGE and OWN)
- **Longer credit history** (cb_preson_cred_hist_length)
- **Stable employment** (longer person_emp_length)
- **Lower loan amounts** relative to income

## Business Applications

### Loan Approval Workflow
1. **Data Collection**: Gather applicant information
2. **Preprocessing**: Apply same transformations as training data
3. **Score Calculation**: Use LightGBM model to predict default probability
4. **Threshold Application**: Apply business-defined acceptance rate strategy
5. **Decision Making**: Approve/reject based on risk tolerance

### Portfolio Management
- **Risk Monitoring**: Track actual vs predicted default rates
- **Strategy Adjustment**: Modify thresholds based on market conditions
- **Expected Loss Tracking**: Monitor portfolio health using PD×EAD×LGD
- **Profitability Analysis**: Regular assessment of net portfolio value

### Expected Loss Management
```python
# Example calculation for a loan
probability_default = model.predict_proba(loan_features)[0][1]
loan_amount = 10000  # EAD
loss_given_default = 0.6  # 60% LGD
expected_loss = probability_default * loan_amount * loss_given_default
```

## Visualization Outputs

The notebook generates several key visualizations:

1. **Correlation Heatmap**: Feature relationships and target correlations
2. **Calibration Curves**: Model reliability for LightGBM and Neural Network
3. **SHAP Summary Plot**: Global feature importance and impact direction
4. **LIME Explanations**: Individual prediction breakdowns (4 random samples)
5. **Business Strategy Charts**: 3-graph layout showing:
   - Acceptance Rate vs Total Expected Loss
   - Acceptance Rate vs Total Expected Revenue  
   - Acceptance Rate vs Net Expected Profit
6. **Expected Loss Distribution**: Portfolio risk analysis histograms

## Files Description

- **`notebook.ipynb`**: Complete analysis pipeline with all modeling and business logic
- **`EDA.ipynb`**: Exploratory data analysis and initial insights
- **`functions.py`**: Utility functions for visualization and data processing
- **`data_models/best_tree_model.pkl`**: Trained LightGBM model (best performer)
- **`data_models/best_nn_model.keras`**: Trained neural network model
- **`data_models/xgb_model.json`**: XGBoost model for missing value imputation
- **`data_models/final_data.csv`**: Fully processed dataset ready for modeling

## Model Performance Summary

| Model | Cross-Val F1 | Test F1 | Key Strengths |
|-------|-------------|---------|---------------|
| LightGBM | ~0.82 | ~0.82 | Best overall performance, well-calibrated |
| Random Forest | ~0.80 | ~0.80 | Stable baseline, good interpretability |
| XGBoost | ~0.81 | ~0.81 | Strong performance, handles missing values |
| Neural Network | ~0.79 | ~0.79 | Captures non-linear patterns, SHAP/LIME ready |
| Ensemble | ~0.81 | ~0.81 | Robust through voting, reduced overfitting |

## Future Enhancements

1. **Advanced Feature Engineering**:
   - Interaction terms between key features
   - Time-based features if temporal data available
   - External data integration (macro-economic indicators)

2. **Model Improvements**:
   - Hyperparameter optimization using Optuna/GridSearch
   - Advanced ensemble methods (stacking, blending)
   - Calibration refinement (Platt scaling, isotonic regression)

3. **Business Integration**:
   - Real-time scoring API development
   - A/B testing framework for strategy optimization
   - Dynamic threshold adjustment based on market conditions

4. **Monitoring and Validation**:
   - Model drift detection
   - Performance degradation alerts
   - Regular backtesting and recalibration

## Contact and Contributions

This project demonstrates end-to-end credit risk modeling including data preprocessing, multiple ML algorithms, model explainability, and business strategy optimization. The code is designed to be production-ready with proper model serialization and comprehensive documentation.

For questions or contributions, please refer to the notebook documentation and inline comments for detailed implementation guidance.
