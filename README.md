# Clinical-Diabetes-Risk-Assessment-with-Interpretable-Machine-Learning

![Diabetes Prediction Visualization](reports/shap_summary.png)  
*Example of model interpretability using SHAP values*

## Overview
This project develops a machine learning system to predict diabetes risk using clinical measurements from the Pima Indians Diabetes dataset. The solution focuses on interpretability and clinical relevance, providing not just predictions but also explanations for each prediction to support medical decision-making.

**Key Metrics**:
- **ROC-AUC**: 0.85
- **Precision/Recall Balance**: 0.76/0.78 (at optimal threshold)
- **F1-Score**: 0.77
- **Accuracy**: 79%

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Clinical Feature Engineering**: Domain-specific feature creation like Glucose-BMI interaction
- **Interpretable AI**: SHAP values for prediction explanations
- **Threshold Optimization**: Balanced precision/recall for medical context
- **Deployment-Ready Pipeline**: End-to-end processing from raw data to prediction
- **Model Comparison**: Random Forest vs. XGBoost vs. Ensemble methods
- **Comprehensive Evaluation**: ROC curves, precision-recall analysis, feature importance

## Installation

1. Clone repository:
```bash
git clone https://github.com/harsh782patel/Clinical-Diabetes-Risk-Assessment-with-Interpretable-Machine-Learning.git
cd Clinical-Diabetes-Risk-Assessment-with-Interpretable-Machine-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Full Analysis
Execute the Jupyter notebook for complete EDA, modeling and evaluation:
```bash
jupyter notebook notebooks/Diabetes_Prediction.ipynb
```

### Making Predictions
Use the trained model to assess diabetes risk:

```python
from src.predict import predict_diabetes

# Sample patient data: [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]
patient_data = [7, 159, 64, 29, 125, 27.4, 0.294, 40]

risk_score = predict_diabetes(patient_data)
print(f"Diabetes probability: {risk_score:.1%}")

# Output: Diabetes probability: 76.0%

```

### Training from Scratch
Retrain the model with custom parameters:
```bash
python src/train.py --n_estimators 300 --max_depth 15
```

## Key Results

### Model Performance Comparison
| Model          | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|----------------|---------|----------|-----------|--------|----------|
| **XGBoost**    | 0.85    | 0.79     | 0.76      | 0.78   | 0.77     |
| Random Forest  | 0.83    | 0.79     | 0.74      | 0.75   | 0.75     |

### Feature Importance
![Feature Importance](reports/feature_importance_comparison.png)  
*Glucose levels, BMI, and Age are the strongest predictors of diabetes risk*

### ROC Curve Comparison
![ROC Curves](reports/roc_comparison.png)  
*XGBoost shows marginally better performance than Random Forest*

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.
