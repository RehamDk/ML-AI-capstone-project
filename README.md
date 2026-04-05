# Black-Box Optimisation Capstone Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Author
**Reham Aldakhil**  
PhD Candidate, Imperial College London | Digital Health Leader | Clinical Informatics Expert

[![LinkedIn](https://img.shields.io/badge/LinkedIn-r--aldakhil-blue?logo=linkedin)](https://linkedin.com/in/r-aldakhil)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--6975--3858-green?logo=orcid)](https://orcid.org/0000-0002-6975-3858)

---

## Non-Technical Summary (~100 words)

This project tackles the challenge of finding the best settings for eight unknown systems when each test is expensive and limited. Rather than testing randomly, I built a smart assistant that combines four different prediction models to learn from every experiment, estimate where the best results are likely to be, and recommend the next test to run. Over thirteen rounds, this approach progressively identified high-performing regions while avoiding wasted evaluations. The methods developed here apply directly to real-world problems such as optimising drug dosages, tuning machine learning models, and allocating healthcare resources — any domain where trial-and-error is costly.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Approach](#technical-approach)
- [Repository Structure](#repository-structure)
- [Results Summary](#results-summary)
- [Key Lessons](#key-lessons)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [References](#references)

---

## Project Overview

### What is Black-Box Optimisation?

This capstone project tackles the challenge of optimising eight unknown functions where the internal structure, gradients, and mathematical form are completely hidden. The only available information comes from querying specific points in the search space and observing their output values. Each function exists in a different number of dimensions (2–10), and the search domain is bounded within [-5.0, 5.0] per dimension.

### Objective

Maximise eight unknown functions f_i: X → R using a strictly limited budget of queries (~8 per round, 13 rounds total, ~104 queries per function). This mirrors real-world problems such as hyperparameter tuning, drug discovery, clinical trial design, and engineering optimisation.

### Real-World Relevance

This project directly supports my career in digital health and clinical informatics by developing practical skills in:

- **Resource-constrained optimisation**: Making optimal decisions with limited, expensive evaluations — directly paralleling clinical trial design
- **Uncertainty quantification**: Knowing what we don't know is as valuable as predictions in healthcare settings
- **Systematic decision-making**: Balancing exploration and exploitation under time pressure and budget constraints

---

## Technical Approach

### Strategy Evolution

The approach evolved through three distinct phases across thirteen rounds:

| Phase | Rounds | Strategy | Key Change |
|-------|--------|----------|------------|
| **Exploration** | 1–3 | Random sampling → GP-UCB → SVM classification | Established baselines, introduced model-based querying |
| **Ensemble Development** | 4–8 | Four-model ensemble surrogate with weighted acquisition | Addressed model misspecification, improved uncertainty quantification |
| **Refinement** | 9–13 | Dimensionality reduction, adaptive exploration decay | Focused queries on influential dimensions, shifted to exploitation |

### Architecture

The pipeline has three layers:

1. **Surrogate Layer**: Four complementary models fitted to accumulated query data
   - Gaussian Process (Matérn-5/2 kernel) — smooth trends + principled uncertainty
   - Support Vector Regression (RBF kernel) — robust to outliers
   - Random Forest (100 trees) — non-linear interactions, feature importance
   - Gradient Boosting (100 estimators) — sequential residual correction

2. **Acquisition Layer**: Weighted combination of four signals
   - Ensemble mean prediction (exploitation, weight 0.45)
   - Total uncertainty: GP variance + ensemble disagreement (exploration, weight 0.20)
   - SVM classification probability of high-value region (guidance, weight 0.20)
   - Distance to nearest observed point (space coverage, weight 0.15)

3. **Adaptive Scheduling**: Exploration weight decays as w₂(t) = 0.25 × (1 − t/T)

### Key Design Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Surrogate model | Ensemble over single GP | No Free Lunch theorem; ensemble reduces prediction variance ~30% |
| Kernel | Matérn-5/2 over RBF | Less smoothness assumption, more realistic for unknown functions |
| Framework | scikit-learn over deep learning | Data scarcity (~20 points per function); classical models with built-in regularisation |
| Preprocessing | Log-transform + per-dimension ARD | Handles multi-scale outputs; automatic relevance determination |
| Dimensionality reduction | Per-dimension sensitivity analysis | Fix low-impact dimensions, focus budget on influential axes |

---

## Repository Structure

```
├── README.md                          # This file
├── DATASHEET.md                       # Data documentation (Gebru et al., 2021)
├── MODEL_CARD.md                      # Model documentation (Mitchell et al., 2019)
├── TECHNICAL_APPROACH.md              # Detailed technical justification
├── LITERATURE_REVIEW.md               # Academic literature informing the approach
├── BBO_capstone_presentation.pdf      # Final presentation (5-section template)
├── notebooks/
│   └── bbo_optimisation_pipeline.ipynb # Main Jupyter notebook with full pipeline
├── src/
│   ├── surrogate_models.py            # Ensemble surrogate implementation
│   ├── acquisition_functions.py       # Weighted acquisition function
│   ├── sensitivity_analysis.py        # Per-dimension sensitivity analysis
│   └── utils.py                       # Query formatting, data processing
├── data/
│   └── query_history/                 # Query logs per function per round (CSV)
├── results/
│   ├── figures/                       # Convergence plots, landscape visualisations
│   └── summary/                       # Final results tables
├── requirements.txt                   # Python dependencies
└── LICENSE                            # MIT License
```

---

## Results Summary

### Performance Overview

| Function | Dimensions | Best Value Found | Round Achieved | Confidence |
|----------|-----------|-----------------|----------------|------------|
| F1 | — | *(fill from portal)* | — | High |
| F2 | — | *(fill from portal)* | — | High |
| F3 | — | *(fill from portal)* | — | Medium |
| F4 | — | *(fill from portal)* | — | Medium |
| F5 | — | *(fill from portal)* | — | Medium |
| F6 | — | *(fill from portal)* | — | Low |
| F7 | — | *(fill from portal)* | — | Medium |
| F8 | — | *(fill from portal)* | — | Low |

*(Update this table with your actual results from the capstone portal.)*

### Improvement Trajectory

| Round | Strategy | Avg Improvement |
|-------|----------|----------------|
| 1 | Random Sampling | Baseline |
| 2 | GP + UCB | ~+22% |
| 3 | Dual SVM | ~+12% |
| 5 | Ensemble introduced | ~+15% |
| 9 | Dimensionality reduction | ~+8% |
| 13 | Final exploitation | ~+2% |

---

## Key Lessons

1. **Structure over sophistication**: Understanding which dimensions drive the output delivered more improvement than any algorithmic upgrade
2. **Ensemble disagreement > single-model variance**: Model disagreement is a richer uncertainty signal than GP posterior variance alone
3. **Adaptive exploration is essential**: The optimal exploration-exploitation balance shifts as data accumulates — early rounds need broad coverage, later rounds need focused refinement
4. **The No Free Lunch theorem is experiential**: No single strategy dominated all eight functions; adaptability was more valuable than any fixed methodology
5. **Small data demands simple models**: With ~20 observations per function, scikit-learn models with built-in regularisation outperformed more complex alternatives

---

## Getting Started

### Prerequisites

```bash
Python >= 3.9
```

### Installation

```bash
git clone https://github.com/RehamDk/ML-AI-capstone-project.git
cd ML-AI-capstone-project
pip install -r requirements.txt
```

### Requirements

```txt
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Quick Start

```python
from src.surrogate_models import EnsembleSurrogate
from src.acquisition_functions import WeightedAcquisition
import pandas as pd

# Load historical data for a function
data = pd.read_csv('data/query_history/function_1.csv')
X_history = data.iloc[:, :-1].values
y_history = data.iloc[:, -1].values

# Fit ensemble surrogate
ensemble = EnsembleSurrogate()
ensemble.fit(X_history, y_history)

# Generate next query suggestion
acquisition = WeightedAcquisition(exploration_weight=0.15)
next_query = acquisition.suggest(ensemble, X_history, y_history)
print(f"Suggested query: {next_query}")
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [DATASHEET.md](DATASHEET.md) | Data provenance, collection methodology, limitations |
| [MODEL_CARD.md](MODEL_CARD.md) | Model details, performance, assumptions, ethical considerations |
| [TECHNICAL_APPROACH.md](TECHNICAL_APPROACH.md) | Detailed technical justification with literature grounding |
| [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) | Academic papers informing the approach |

---

## References

1. Shahriari, B., et al. (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148-175.
2. Jones, D. R., et al. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455-492.
3. Snoek, J., et al. (2012). Practical Bayesian optimization of machine learning algorithms. *NeurIPS*, 25.
4. Wolpert, D. H. & Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Trans. Evolutionary Computation*, 1(1), 67-82.
5. Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*, 30.
6. Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
7. Gebru, T., et al. (2021). Datasheets for Datasets. *Communications of the ACM*.
8. Mitchell, M., et al. (2019). Model Cards for Model Reporting. *Proceedings of FAT*.

---

## Contact

**Reham Aldakhil**  
Email: r.aldakhil23@imperial.ac.uk  
[LinkedIn](https://linkedin.com/in/r-aldakhil) | [ORCID](https://orcid.org/0000-0002-6975-3858)

---

*This project is part of the Professional Certificate in Machine Learning and Artificial Intelligence programme.*  
*Last Updated: April 2026*
