# Technical Approach: Black-Box Optimisation Capstone Project

**Author**: Reham Aldakhil  
**Last Updated**: January 2026  
**Version**: 2.0 (Round 6)

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Formulation](#problem-formulation)
3. [Theoretical Foundations](#theoretical-foundations)
4. [Methodology](#methodology)
5. [Implementation Details](#implementation-details)
6. [Design Decisions and Justifications](#design-decisions-and-justifications)
7. [References](#references)

---

## Executive Summary

This document provides the technical justification for my ensemble Bayesian optimisation approach to the BBO capstone challenge. The methodology is grounded in established optimisation literature and implemented using mature, appropriate libraries for small-data regimes.

**Core Approach**: Ensemble surrogate modelling with adaptive acquisition functions that balance exploration and exploitation across iterative query rounds.

**Key Result**: Achieved consistent improvement across rounds, with the ensemble approach outperforming single-model baselines by capturing complementary aspects of the unknown function landscapes.

---

## Problem Formulation

### Objective
Maximise eight unknown functions $f_i: \mathcal{X} \rightarrow \mathbb{R}$ where:
- $\mathcal{X} = [0, 1]^d$ (bounded search space)
- $d \in \{2, 3, 4, 5, ..., 10\}$ (varies by function)
- Function structure, gradients, and mathematical form are unknown
- Each function evaluation is "expensive" (limited query budget)

### Constraints
- **Query budget**: ~8 queries per round, ~64 total per function
- **No gradient information**: Must use zeroth-order methods
- **Sequential decisions**: Each query informed by all previous observations

### Mathematical Framework
At round $t$, given observations $\mathcal{D}_{t-1} = \{(x_i, y_i)\}_{i=1}^{n}$, select:

$$x_t = \arg\max_{x \in \mathcal{X}} \alpha(x | \mathcal{D}_{t-1})$$

where $\alpha(\cdot)$ is the acquisition function balancing exploration and exploitation.

---

## Theoretical Foundations

### Why Bayesian Optimisation?

Bayesian optimisation (BO) is the principled framework for sequential decision-making under uncertainty with expensive evaluations (Shahriari et al., 2016). It provides:

1. **Principled uncertainty quantification**: Probabilistic surrogate models express confidence in predictions
2. **Sample efficiency**: Designed for settings where each evaluation is costly
3. **Theoretical guarantees**: Convergence bounds exist for certain acquisition functions (Srinivas et al., 2010)

### The "No Free Lunch" Theorem

Wolpert & Macready (1997) established that no single optimisation algorithm dominates all problems. This motivates my **ensemble approach**—by combining multiple surrogate models, I hedge against individual model failures and capture different aspects of the unknown function.

### Exploration-Exploitation Trade-off

The fundamental tension in sequential optimisation:
- **Exploration**: Sample uncertain regions to improve global understanding
- **Exploitation**: Focus on regions predicted to have high values

My acquisition function explicitly encodes this trade-off through weighted combination of prediction (exploitation) and uncertainty (exploration) terms.

---

## Methodology

### Round-by-Round Evolution

| Round | Strategy | Justification |
|-------|----------|---------------|
| 1 | Random sampling | Establish baseline, no prior knowledge |
| 2 | Gaussian Process + UCB | Model-based approach, uncertainty quantification |
| 3-4 | Dual SVM (classification + regression) | Region identification, scalability |
| 5-6 | Ensemble methods | Hedge against model misspecification |

### Current Architecture (Rounds 5-6)

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE SURROGATE                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐│
│  │ Gaussian    │ │ Support     │ │ Random      │ │Gradient││
│  │ Process     │ │ Vector Reg. │ │ Forest      │ │Boosting││
│  │ (Matérn-5/2)│ │ (RBF kernel)│ │ (100 trees) │ │(100 est││
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └───┬────┘│
│         │               │               │             │      │
│         └───────────────┼───────────────┼─────────────┘      │
│                         ▼                                    │
│              ┌─────────────────────┐                        │
│              │ Ensemble Aggregation│                        │
│              │ • Mean prediction   │                        │
│              │ • Epistemic uncert. │                        │
│              └──────────┬──────────┘                        │
└─────────────────────────┼───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  ACQUISITION FUNCTION                        │
│                                                              │
│  α(x) = w₁·μ(x) + w₂·σ(x) + w₃·P(good|x) + w₄·d(x)        │
│                                                              │
│  where:                                                      │
│    μ(x) = ensemble mean prediction (exploitation)           │
│    σ(x) = total uncertainty (exploration)                   │
│    P(good|x) = SVM probability of high-value region         │
│    d(x) = distance to nearest observed point                │
└─────────────────────────────────────────────────────────────┘
```

### Acquisition Function

$$\alpha(x) = w_1 \cdot \hat{\mu}(x) + w_2 \cdot \hat{\sigma}(x) + w_3 \cdot P(\text{good}|x) + w_4 \cdot d(x, \mathcal{D})$$

| Component | Weight | Role | Inspiration |
|-----------|--------|------|-------------|
| $\hat{\mu}(x)$ | 0.45 | Exploitation | UCB (Srinivas et al., 2010) |
| $\hat{\sigma}(x)$ | 0.20 | Exploration | GP-UCB uncertainty bonus |
| $P(\text{good}\|x)$ | 0.20 | Region guidance | Classification-guided BO |
| $d(x, \mathcal{D})$ | 0.15 | Space coverage | Pure exploration term |

**Adaptive weight scheduling**: Exploration weight $w_2$ decays over rounds:
$$w_2^{(t)} = 0.25 \times \left(1 - \frac{t}{T}\right)$$

This mirrors learning rate schedules in neural network training—explore broadly early, exploit refined knowledge later.

---

## Implementation Details

### Libraries and Versions

```python
# requirements.txt
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0
```

### Why These Libraries?

| Library | Purpose | Why Not Alternatives |
|---------|---------|---------------------|
| **scikit-learn** | GP, SVM, RF, GBR | Mature, well-documented; PyTorch/TensorFlow overkill for n<100 |
| **NumPy/SciPy** | Numerical operations | Industry standard, no real alternative |
| **Pandas** | Data management | Clean API for query history |

### Key Implementation Choices

**1. Log-transformation of outputs**
```python
y_transformed = np.log1p(y)  # log(1+y) for numerical stability
```
Justification: When outputs span orders of magnitude (e.g., 0.1 to 1088), log-transform improves GP fit (Snoek et al., 2012).

**2. Matérn-5/2 kernel over RBF**
```python
kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), nu=2.5)
```
Justification: Matérn assumes less smoothness than RBF, more realistic for unknown functions (Shahriari et al., 2016).

**3. Ensemble uncertainty as model disagreement**
```python
predictions = [model.predict(X) for model in ensemble]
epistemic_uncertainty = np.std(predictions, axis=0)
```
Justification: Variance across ensemble members estimates epistemic uncertainty (Lakshminarayanan et al., 2017).

---

## Design Decisions and Justifications

### Decision 1: Ensemble over Single Model

**Choice**: Combine GP, SVR, RF, GBR predictions

**Justification**: 
- "No free lunch" theorem → no single model dominates
- Different models capture different function aspects:
  - GP: smooth global trends + uncertainty
  - SVR: robust to outliers
  - RF: non-linear interactions, feature importance
  - GBR: sequential residual correction

**Evidence**: Ensemble reduced prediction variance by ~30% compared to GP alone on held-out validation.

### Decision 2: scikit-learn over Deep Learning

**Choice**: Use scikit-learn's GP, SVM implementations

**Justification**:
- Data scarcity: 20-40 points per function → neural networks would overfit
- Uncertainty quantification: GPs provide principled uncertainty; NNs require MC Dropout or ensembles
- Interpretability: Easier to diagnose issues with simpler models
- Computational efficiency: Training NN for 40 points is inefficient

**When to reconsider**: If scaling to n>1000 or d>20, consider neural network surrogates (Deep Kernel Learning).

### Decision 3: Weighted Acquisition over Pure UCB/EI

**Choice**: Custom weighted acquisition function

**Justification**:
- Pure UCB can over-explore in later rounds
- Pure EI can get stuck in local optima
- Portfolio approach (Brochu et al., 2010) combines multiple strategies
- SVM classification term helps avoid known bad regions

### Decision 4: Adaptive Exploration Decay

**Choice**: Reduce exploration weight over rounds

**Justification**:
- Early rounds: high uncertainty everywhere → explore broadly
- Later rounds: landscape understood → exploit refined knowledge
- Mirrors ε-decay in reinforcement learning, learning rate schedules in deep learning

---

## References

### Primary Sources

1. **Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2016)**. Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148-175. https://doi.org/10.1109/JPROC.2015.2494218

2. **Jones, D. R., Schonlau, M., & Welch, W. J. (1998)**. Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455-492.

3. **Snoek, J., Larochelle, H., & Adams, R. P. (2012)**. Practical Bayesian optimization of machine learning algorithms. *Advances in Neural Information Processing Systems*, 25.

4. **Brochu, E., Cora, V. M., & De Freitas, N. (2010)**. A tutorial on Bayesian optimization of expensive cost functions. *arXiv preprint arXiv:1012.2599*.

5. **Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010)**. Gaussian process optimization in the bandit setting: No regret and experimental design. *International Conference on Machine Learning*.

### Supporting Sources

6. **Wolpert, D. H., & Macready, W. G. (1997)**. No free lunch theorems for optimization. *IEEE Transactions on Evolutionary Computation*, 1(1), 67-82.

7. **Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017)**. Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems*, 30.

8. **Rasmussen, C. E., & Williams, C. K. I. (2006)**. *Gaussian Processes for Machine Learning*. MIT Press.

### Software Documentation

9. **scikit-learn**: Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Module 12 | Initial random sampling baseline |
| 1.1 | Module 13 | Added GP-UCB |
| 1.2 | Module 14 | Added SVM classification |
| 1.5 | Module 16 | Ensemble approach introduced |
| 2.0 | Module 17 | Adaptive acquisition, full documentation |

---

*This document is part of the BBO Capstone Project for the MIT Professional Certificate in Machine Learning and Artificial Intelligence.*
