# Credit Risk Analysis and Modeling

This repository contains a comprehensive credit risk analysis and machine learning modeling project that predicts loan defaults using various algorithms and provides business insights for loan approval strategies.

## Project Overview

The project analyzes credit risk data to build predictive models that can help financial institutions make informed loan approval decisions. It includes data preprocessing, feature engineering, multiple machine learning models, model explainability, and business strategy optimization.

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
- **Outlier Removal**: Removes extreme values based on domain knowledge
- **One-Hot Encoding**: Categorical variables (home ownership, loan intent)
- **Ordinal Encoding**: Loan grades (A=1 through G=7)
- **Binary Encoding**: Historical default flag

### 2. Missing Value Imputation
- **Employment Length**: Mean imputation
- **Interest Rate**: XGBoost regression model prediction

### 3. Model Development

#### Tree-Based Models
- **Random Forest**: Ensemble baseline model
- **LightGBM**: Best performing tree model
- **XGBoost**: Alternative gradient boosting
- **Ensemble**: Voting classifier combining all three

#### Neural Network
- **Architecture**: 5-layer deep network (128→64→32→16→1)
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Regularization**: Early stopping
- **Scaling**: RobustScaler for feature normalization

### 4. Model Evaluation
- **Cross-Validation**: 5-fold stratified CV
- **Metrics**: F1-score, Precision, Recall, Accuracy
- **Calibration Curves**: Model reliability assessment

### 5. Model Explainability

#### SHAP Analysis
- **Global Importance**: Feature ranking across all predictions
- **Local Explanations**: Individual prediction breakdowns
- **Visualization**: Summary plots and feature importance charts

#### LIME Analysis
- **Local Interpretability**: Instance-level explanations
- **Feature Contributions**: Positive/negative impact visualization
- **Random Sample Analysis**: Detailed breakdown of selected cases

### 6. Business Strategy Optimization
- **Acceptance Rate Analysis**: 10% to 100% acceptance rates
- **Risk-Return Trade-off**: Bad rate vs portfolio value
- **Expected Loss Calculation**: PD × EAD × LGD formula
- **Optimal Strategy**: Maximizing net portfolio value

## Key Results

### Model Performance
- **Best Model**: LightGBM classifier
- **F1-Score**: ~0.65 (cross-validation)
- **Calibration**: Well-calibrated probability predictions

### Feature Importance
Top predictive features (via SHAP analysis):
1. Loan grade
2. Interest rate
3. Loan amount
4. Income percentage
5. Employment length

### Business Insights
- **Optimal Acceptance Rate**: Varies based on risk tolerance
- **Expected Loss Rate**: ~X% of total portfolio value
- **Strategy Recommendations**: Risk-based pricing and thresholds

## Usage

### Requirements
```python
pip install pandas numpy scikit-learn lightgbm xgboost tensorflow
pip install shap lime matplotlib seaborn joblib
```

### Running the Analysis
1. **Data Preprocessing**: Execute cells 1-30 in `notebook.ipynb`
2. **Model Training**: Run model sections (cells 31-60)
3. **Model Evaluation**: Execute evaluation cells (61-80)
4. **Explainability**: Run SHAP/LIME analysis (cells 81-90)
5. **Business Strategy**: Execute strategy optimization (cells 91-100)

### Loading Saved Models
```python
import joblib
from tensorflow import keras

# Load best tree model
best_tree_model = joblib.load('data_models/best_tree_model.pkl')

# Load neural network
best_nn_model = keras.models.load_model('data_models/best_nn_model.keras')
```

## Model Interpretation

### Risk Factors (Increase Default Probability)
- Higher loan grades (E, F, G)
- High loan-to-income ratio
- History of defaults
- Shorter employment history

### Protective Factors (Decrease Default Probability)
- Lower loan grades (A, B, C)
- Home ownership
- Longer credit history
- Stable employment

## Business Applications

### Loan Approval Process
1. **Score Calculation**: Use model to predict default probability
2. **Threshold Setting**: Based on acceptance rate strategy
3. **Risk-Based Pricing**: Adjust interest rates by risk level
4. **Portfolio Monitoring**: Track actual vs predicted performance

### Expected Loss Management
- **PD**: Probability of Default from model
- **EAD**: Exposure at Default (loan amount)
- **LGD**: Loss Given Default (assumed 60%)
- **Expected Loss**: PD × EAD × LGD

## Files Description

- `notebook.ipynb`: Complete analysis pipeline
- `EDA.ipynb`: Exploratory data analysis
- `functions.py`: Helper functions for visualization and analysis
- `data_models/`: Trained models and processed datasets

## Future Enhancements

1. **Feature Engineering**: Create additional predictive features
2. **Advanced Models**: Try ensemble methods, neural architecture search
3. **Calibration**: Implement Platt scaling or isotonic regression
4. **Monitoring**: Build model drift detection
5. **API Development**: Deploy model as REST API

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please reach out via GitHub issues.
