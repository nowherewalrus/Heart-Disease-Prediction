# Heart Disease Classification Project

## 📋 Project Overview
This project implements machine learning classification algorithms to detect heart attack risk based on patient health data. The system analyzes clinical parameters to classify patients into high-risk and low-risk categories, providing valuable insights for early intervention and preventive healthcare.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Objective
To develop and compare multiple machine learning models for accurate heart disease risk prediction, with a focus on minimizing false negatives (patients incorrectly classified as low risk).

## 📊 Dataset
The project uses the Heart Disease dataset containing 303 patient records with 14 clinical features:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Gender (1=male, 0=female) | Categorical |
| cp | Chest pain type (0-3) | Categorical |
| trtbps | Resting blood pressure (mm Hg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl (1=true, 0=false) | Categorical |
| restecg | Resting electrocardiographic results (0-2) | Categorical |
| thalachh | Maximum heart rate achieved | Numerical |
| exng | Exercise induced angina (1=yes, 0=no) | Categorical |
| oldpeak | ST depression induced by exercise | Numerical |
| slp | Slope of peak exercise ST segment | Categorical |
| caa | Number of major vessels colored by fluoroscopy (0-3) | Categorical |
| thall | Thalassemia (1-3) | Categorical |
| output | Target variable (1=heart disease, 0=no heart disease) | Binary |

## 🏗️ Project Structure
```
Heart-Risk-Prediction/
│
├── heart.csv                    # Dataset file
├── Heart-Risk-Prediction.ipynb  # Main Jupyter Notebook
├── README.md                    # This file
├── requirements.txt             # Dependencies
└── images/                      # Output visualizations (optional)
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/nowherewalrus/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction.git
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Requirements
Create a `requirements.txt` file with:
```
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
matplotlib==3.4.3
scipy==1.7.1
jupyter==1.0.0
```

## 🚀 Usage

### Run the Jupyter Notebook
```bash
jupyter notebook heart_disease_classification.ipynb
```

### Or execute as Python script
```bash
python heart_disease_classification.py
```

## 📈 Methodology

### 1. Data Preprocessing
- **Feature Selection**: Correlation-based selection (features with ≥0.2 correlation to target)
- **Outlier Removal**: Z-score method (|Z| < 3)
- **Normalization**: StandardScaler for feature scaling

### 2. Models Implemented
Three classification algorithms were implemented and compared:

#### **a. Support Vector Machine (SVM)**
- Kernel: RBF (Radial Basis Function)
- Purpose: Effective for non-linear decision boundaries

#### **b. Logistic Regression**
- Regularization: C=0.01 (strong regularization)
- Solver: liblinear (suitable for small datasets)

#### **c. Decision Tree**
- Criterion: Entropy (information gain)
- Pruning: max_depth=5, min_samples_split=10, min_samples_leaf=5

### 3. Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Jaccard Score**: Similarity coefficient between prediction and ground truth
- **Confusion Matrix**: Detailed breakdown of predictions

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | F1 Score | Jaccard (Class 0) | Jaccard (Class 1) | False Negatives | False Positives |
|-------|----------|----------|-------------------|-------------------|-----------------|-----------------|
| Support Vector Machine | 82.76% | 82.76% | 64.29% | 75.00% | 5 | 5 |
| Logistic Regression | 82.76% | 82.76% | 64.29% | 75.00% | 5 | 5 |
| Decision Tree | 74.14% | 74.35% | 53.13% | 63.41% | 9 | 6 |

### Confusion Matrices
Each model's confusion matrix provides insights into prediction patterns:
- **True Positives (TP)**: Correctly identified high-risk patients
- **True Negatives (TN)**: Correctly identified low-risk patients
- **False Positives (FP)**: Low-risk patients flagged as high-risk (less critical)
- **False Negatives (FN)**: High-risk patients missed (clinically critical)

## 🏥 Clinical Implications

### Critical Findings
1. **False Negatives are Critical**: Missing high-risk patients is more dangerous than false alarms
2. **False Positives are Acceptable**: Additional testing for low-risk patients is preferable to missing high-risk cases
3. **Model Selection**: SVM and Logistic Regression show best balance between sensitivity and specificity

### Recommended Use
- **Primary Screening**: Use SVM or Logistic Regression for initial risk assessment
- **Secondary Verification**: Flagged cases should receive additional medical evaluation
- **Regular Updates**: Models should be retrained with new clinical data periodically

## 🔧 Code Structure

### Main Components
1. **Data Loading & Exploration** (`load_and_explore_data()`)
2. **Feature Engineering** (`feature_selection_correlation()`)
3. **Data Preprocessing** (`remove_outliers_zscore()`, `normalize_features()`)
4. **Model Training** (`train_svm()`, `train_logistic_regression()`, `train_decision_tree()`)
5. **Evaluation** (`evaluate_model()`, `plot_confusion_matrix()`)
6. **Comparison** (`compare_models()`)

### Key Functions
```python
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix'):
    """Visualizes confusion matrix with annotations"""
    # Implementation details...

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation with multiple metrics"""
    # Implementation details...
```

## 🚀 Future Improvements

### Technical Enhancements
- [ ] Implement cross-validation for more robust evaluation
- [ ] Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Experiment with ensemble methods (Random Forest, Gradient Boosting)
- [ ] Develop neural network approach with TensorFlow/PyTorch
- [ ] Create web application interface using Flask/Streamlit

### Clinical Enhancements
- [ ] Incorporate additional clinical features
- [ ] Add explainable AI (SHAP/LIME) for model interpretability
- [ ] Develop risk score calculator
- [ ] Create patient-specific recommendation system

## 📚 References

1. UCI Machine Learning Repository: Heart Disease Dataset
2. American Heart Association: Heart Disease Risk Factors
3. Scikit-learn Documentation
4. Journal of Medical Systems: ML in Cardiology

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Dataset providers: UCI Machine Learning Repository
- Scikit-learn development team
- Medical professionals for domain expertise validation

## 📧 Contact

For questions or suggestions, please contact:
- **Your Name** - pkhaghani916@gmail.com
- **Project Link**: https://github.com/nowherewalrus/Heart-Disease-Prediction.git

---

⭐ **If you find this project useful, please give it a star!** ⭐
