# Literature Review: Bayesian Optimisation for Black-Box Functions

**Author**: Reham Aldakhil  
**Last Updated**: January 2026

---

## Overview

This document summarises the key academic literature informing my BBO capstone project approach. Each paper is briefly described with its main contribution and how I applied its insights.

---

## Foundational Papers

### 1. Shahriari et al. (2016) - "Taking the Human Out of the Loop"

**Citation**: Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148-175.

**Summary**: Comprehensive review of Bayesian optimisation covering surrogate models, acquisition functions, and practical considerations.

**Key Insights**:
- Gaussian Processes as default surrogate model for smooth functions
- Matérn kernels often outperform RBF for real-world functions
- Acquisition functions encode exploration-exploitation trade-off

**My Application**: 
- Adopted Matérn-5/2 kernel instead of RBF
- Used UCB-style acquisition as baseline
- Followed best practices for GP hyperparameter optimisation

---

### 2. Jones et al. (1998) - "Efficient Global Optimization"

**Citation**: Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455-492.

**Summary**: Introduced the EGO algorithm and Expected Improvement (EI) acquisition function.

**Key Insights**:
- EI balances exploration and exploitation naturally
- Closed-form solution exists for GP surrogates
- Sequential sampling dramatically outperforms space-filling designs

**My Application**:
- Implemented EI as alternative acquisition for comparison
- Validated that model-based approach beats random sampling

**EI Formula**:
$$EI(x) = \mathbb{E}[\max(0, f(x) - f^*)] = (μ(x) - f^*) Φ(Z) + σ(x) φ(Z)$$

where $Z = \frac{μ(x) - f^*}{σ(x)}$

---

### 3. Snoek et al. (2012) - "Practical Bayesian Optimization"

**Citation**: Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. *Advances in Neural Information Processing Systems*, 25.

**Summary**: Demonstrated practical effectiveness of BO for hyperparameter tuning, introduced several practical improvements.

**Key Insights**:
- Log-transform outputs spanning multiple orders of magnitude
- Automatic relevance determination (ARD) for feature importance
- BO significantly outperforms random/grid search with limited budget

**My Application**:
- Applied log-transformation: `y_log = np.log1p(y)` for functions with large output ranges
- Used ARD-style separate length scales per dimension

---

### 4. Brochu et al. (2010) - "A Tutorial on Bayesian Optimization"

**Citation**: Brochu, E., Cora, V. M., & De Freitas, N. (2010). A tutorial on Bayesian optimization of expensive cost functions. *arXiv preprint arXiv:1012.2599*.

**Summary**: Accessible tutorial covering BO fundamentals, acquisition functions, and practical advice.

**Key Insights**:
- Portfolio of acquisition functions can outperform single acquisition
- Trade-off between acquisition function complexity and optimisation difficulty
- Importance of multi-start optimisation for acquisition maximisation

**My Application**:
- Designed weighted portfolio acquisition combining UCB, classification probability, and distance
- Used 100,000 random candidates with multi-start optimisation

---

## Ensemble and Uncertainty Quantification

### 5. Lakshminarayanan et al. (2017) - "Deep Ensembles"

**Citation**: Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems*, 30.

**Summary**: Showed that ensembles of neural networks provide well-calibrated uncertainty estimates.

**Key Insights**:
- Ensemble disagreement captures epistemic uncertainty
- Simple to implement, often outperforms complex Bayesian methods
- Works across different model types, not just neural networks

**My Application**:
- Used variance across GP, SVR, RF, GBR predictions as epistemic uncertainty
- Combined with GP's aleatoric uncertainty for total uncertainty estimate

```python
epistemic_uncertainty = np.std([gp_pred, svr_pred, rf_pred, gbr_pred], axis=0)
total_uncertainty = np.sqrt(gp_std**2 + epistemic_uncertainty**2)
```

---

### 6. Wolpert & Macready (1997) - "No Free Lunch Theorems"

**Citation**: Wolpert, D. H., & Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Transactions on Evolutionary Computation*, 1(1), 67-82.

**Summary**: Proved that no single optimisation algorithm dominates all problems when averaged over all possible functions.

**Key Insight**: Ensemble approaches hedge against model misspecification by combining multiple algorithms.

**My Application**: Primary motivation for ensemble approach rather than relying on single surrogate model.

---

## Gaussian Processes

### 7. Rasmussen & Williams (2006) - "Gaussian Processes for Machine Learning"

**Citation**: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

**Summary**: Definitive textbook on Gaussian Processes covering theory and practice.

**Key Insights**:
- GP posterior provides mean (prediction) and variance (uncertainty)
- Kernel choice encodes assumptions about function properties
- Marginal likelihood for hyperparameter optimisation

**My Application**:
- GP as primary surrogate model
- Multiple kernel restarts for robust hyperparameter estimation
- Understanding of GP limitations (cubic scaling, smoothness assumptions)

---

## Advanced Topics (For Future Reference)

### 8. Eriksson et al. (2019) - "TuRBO"

**Citation**: Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019). Scalable global optimization via local Bayesian optimization. *Advances in Neural Information Processing Systems*, 32.

**Summary**: Trust Region Bayesian Optimization for high-dimensional problems.

**Potential Application**: If scaling to d>20 dimensions, TuRBO's local search approach would be relevant.

---

### 9. Frazier (2018) - "A Tutorial on Bayesian Optimization"

**Citation**: Frazier, P. I. (2018). A tutorial on Bayesian optimization. *arXiv preprint arXiv:1807.02811*.

**Summary**: More recent comprehensive tutorial with modern perspectives.

**Potential Application**: Reference for future refinements and best practices updates.

---

## Summary Table

| Paper | Year | Key Contribution | Applied In |
|-------|------|------------------|------------|
| Shahriari et al. | 2016 | BO comprehensive review | Kernel choice, acquisition design |
| Jones et al. | 1998 | EGO algorithm, EI | Expected Improvement implementation |
| Snoek et al. | 2012 | Practical BO | Log-transform, ARD |
| Brochu et al. | 2010 | BO tutorial, portfolio | Weighted acquisition function |
| Lakshminarayanan et al. | 2017 | Ensemble uncertainty | Epistemic uncertainty via disagreement |
| Wolpert & Macready | 1997 | No free lunch | Motivation for ensemble |
| Rasmussen & Williams | 2006 | GP textbook | GP implementation details |

---

## BibTeX

```bibtex
@article{shahriari2016taking,
  title={Taking the human out of the loop: A review of Bayesian optimization},
  author={Shahriari, Bobak and Swersky, Kevin and Wang, Ziyu and Adams, Ryan P and De Freitas, Nando},
  journal={Proceedings of the IEEE},
  volume={104},
  number={1},
  pages={148--175},
  year={2016}
}

@article{jones1998efficient,
  title={Efficient global optimization of expensive black-box functions},
  author={Jones, Donald R and Schonlau, Matthias and Welch, William J},
  journal={Journal of Global optimization},
  volume={13},
  number={4},
  pages={455--492},
  year={1998}
}

@inproceedings{snoek2012practical,
  title={Practical Bayesian optimization of machine learning algorithms},
  author={Snoek, Jasper and Larochelle, Hugo and Adams, Ryan P},
  booktitle={Advances in Neural Information Processing Systems},
  volume={25},
  year={2012}
}

@article{brochu2010tutorial,
  title={A tutorial on Bayesian optimization of expensive cost functions},
  author={Brochu, Eric and Cora, Vlad M and De Freitas, Nando},
  journal={arXiv preprint arXiv:1012.2599},
  year={2010}
}

@inproceedings{lakshminarayanan2017simple,
  title={Simple and scalable predictive uncertainty estimation using deep ensembles},
  author={Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}

@article{wolpert1997no,
  title={No free lunch theorems for optimization},
  author={Wolpert, David H and Macready, William G},
  journal={IEEE Transactions on Evolutionary Computation},
  volume={1},
  number={1},
  pages={67--82},
  year={1997}
}

@book{rasmussen2006gaussian,
  title={Gaussian Processes for Machine Learning},
  author={Rasmussen, Carl Edward and Williams, Christopher KI},
  year={2006},
  publisher={MIT Press}
}
```

---

*This literature review supports the BBO Capstone Project for the MIT Professional Certificate in Machine Learning and Artificial Intelligence.*
