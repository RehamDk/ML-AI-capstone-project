# ML/AI Capstone Project: Virtual vs In-Person Consultation Classifier for Type 2 Diabetes Patients

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

## Author
**Reham Aldakhil**  
PhD Candidate, Imperial College London | Digital Health Leader | Clinical Informatics Expert

[![LinkedIn](https://img.shields.io/badge/LinkedIn-r--aldakhil-blue?logo=linkedin)](https://linkedin.com/in/r-aldakhil)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--6975--3858-green?logo=orcid)](https://orcid.org/0000-0002-6975-3858)

---

## Project Overview

This project develops a machine learning model to predict the optimal consultation mode (virtual vs in-person) for patients with Type 2 Diabetes. The goal is to optimize healthcare resource allocation while maintaining quality of care.

### Problem Statement
Healthcare systems face challenges in efficiently allocating consultation modes for diabetes patients. Currently, decisions are made inconsistently, leading to:
- Unnecessary in-person visits for patients suitable for virtual care
- Virtual consultations for patients requiring physical examination
- Suboptimal resource utilization

### Objective
Build a supervised classification model that recommends the most appropriate consultation mode based on patient demographics, clinical characteristics, and historical outcomes.

---

## Research Context

This project extends my PhD research at Imperial College London on the **Impact of Virtual Consultations on Quality of Care for Patients with Type 2 Diabetes**, published in the *Journal of Diabetes Science and Technology* (2025).

**Related Publication:**  
Aldakhil, R., Greenfield, G., Lammila-Escalera, E., et al. (2025). *The Impact of Virtual Consultations on Quality of Care for Patients with Type 2 Diabetes: A Systematic Review and Meta-analysis.* [DOI: 10.1177/19322968251316585](https://doi.org/10.1177/19322968251316585)

---

## Dataset

### Data Source
Electronic Health Records (EHR) from healthcare facilities offering both virtual and in-person diabetes consultations.

### Features (Input Variables)

| Category | Features |
|----------|----------|
| **Demographics** | Age, gender, location (urban/rural), employment status |
| **Clinical** | HbA1c levels, BMI, comorbidities, medication complexity, years since diagnosis |
| **Technology Access** | Internet availability, digital literacy score |
| **Historical** | Appointment attendance rate, complications history |

### Target Variable (Output)
- **Consultation Mode**: Binary classification (Virtual / In-Person)

---

## Technologies Used

- **Programming**: Python 3.9+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Statistical Analysis**: SciPy, Statsmodels
- **Environment**: Jupyter Notebooks

---

## Repository Structure

```
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── utils.py
├── results/
│   ├── figures/
│   └── models/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ML Pipeline

1. **Define Purpose**: Optimize consultation mode allocation for T2D patients
2. **Data Collection**: Extract EHR data with ethical approval
3. **Data Exploration**: Analyze distributions, identify patterns
4. **Preprocessing**: Handle missing values, outliers, encode categorical variables
5. **Feature Engineering**: Create clinically relevant features
6. **Data Splitting**: Training (70%), Validation (15%), Test (15%)
7. **Model Selection**: Logistic Regression (baseline), Random Forest, XGBoost
8. **Training & Tuning**: Hyperparameter optimization via cross-validation
9. **Evaluation**: Accuracy, Sensitivity, Specificity, AUC-ROC
10. **Deployment**: Clinical decision support integration

---

## Expected Outcomes

- Classification model with >80% accuracy
- Interpretable feature importance for clinical adoption
- Recommendations for healthcare policy on virtual care allocation

---

## Getting Started

### Prerequisites
```bash
python >= 3.9
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/[username]/diabetes-consultation-classifier.git
cd diabetes-consultation-classifier
pip install -r requirements.txt
```

### Usage
```python
from src.model import ConsultationClassifier

# Load trained model
model = ConsultationClassifier.load('results/models/best_model.pkl')

# Predict consultation mode
prediction = model.predict(patient_data)
```


## 📧 Contact

**Reham Aldakhil**  
📧 r.aldakhil23@imperial.ac.uk  
🔗 [LinkedIn](https://linkedin.com/in/r-aldakhil) | [ORCID](https://orcid.org/0000-0002-6975-3858)

---

*This project is part of the Professional Certificate in Machine Learning and Artificial Intelligence programme.*
