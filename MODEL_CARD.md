# Model Card: Ensemble Bayesian Black-Box Optimisation

**Author**: Reham Aldakhil  
**Last Updated**: April 2026

*Following the framework proposed by Mitchell et al. (2019), "Model Cards for Model Reporting," Proceedings of FAT*.*

---

## 1. Model Details

| Field | Details |
|-------|---------|
| **Name** | Ensemble Bayesian Black-Box Optimiser (EBBBO) |
| **Type** | Ensemble surrogate model with adaptive acquisition function |
| **Version** | 3.0 (Final, after 13 rounds) |
| **Author** | Reham Aldakhil |
| **Date** | April 2026 |
| **Framework** | scikit-learn 1.0+, NumPy, SciPy |
| **Licence** | MIT |

### Model Components

The optimiser comprises four surrogate models combined into an ensemble:

| Component | Implementation | Role |
|-----------|---------------|------|
| Gaussian Process | `GaussianProcessRegressor` with Matérn-5/2 kernel | Smooth global trends + principled uncertainty |
| Support Vector Regression | `SVR` with RBF kernel | Robust to outliers |
| Random Forest | `RandomForestRegressor` (100 trees) | Non-linear interactions, feature importance |
| Gradient Boosting | `GradientBoostingRegressor` (100 estimators) | Sequential residual correction |

### Acquisition Function

The acquisition function scores candidate query points using:

α(x) = w₁·μ(x) + w₂·σ(x) + w₃·P(good|x) + w₄·d(x,D)

| Component | Weight | Role |
|-----------|--------|------|
| μ(x): Ensemble mean | 0.45 | Exploitation |
| σ(x): Total uncertainty | 0.20 (decaying) | Exploration |
| P(good\|x): SVM probability | 0.20 | Region guidance |
| d(x,D): Distance to data | 0.15 | Space coverage |

---

## 2. Intended Use

**Suitable for:**
- Low-budget black-box optimisation with severely limited function evaluations
- Sequential decision-making under uncertainty where each evaluation is expensive
- Educational settings demonstrating Bayesian optimisation principles
- Problems with continuous search spaces in 2–10 dimensions

**Not suitable for:**
- High-dimensional problems (d > 15) without modification — the curse of dimensionality degrades surrogate accuracy
- Problems requiring convergence guarantees or probabilistic bounds on optimality
- Real-time applications — the ensemble requires retraining after each observation
- Discrete or combinatorial optimisation problems

---

## 3. Training Data

The surrogate models are trained on the accumulated query-response dataset described in DATASHEET.md:

- **Size**: ~104 observations per function (8 queries × 13 rounds)
- **Input space**: Continuous, bounded within [-5.0, 5.0] per dimension
- **Output**: Scalar function value
- **Preprocessing**: Log-transformation for multi-scale outputs; input normalisation to [0, 1]
- **Important bias**: Query points are not uniformly distributed — later rounds are concentrated near predicted optima

---

## 4. Performance

### Strategy Evolution

| Phase | Rounds | Approach | Avg Improvement |
|-------|--------|----------|----------------|
| Exploration | 1–3 | Random → GP-UCB → SVM | ~22% (Round 2) |
| Ensemble | 4–8 | Four-model ensemble | ~15% (Round 5) |
| Refinement | 9–13 | Dimensionality reduction + exploitation | ~8% (Round 9) |

### Confidence Assessment

- **High confidence**: Functions where the best value was confirmed by nearby queries showing worse values in all directions
- **Medium confidence**: Trend suggests proximity to an optimum, but convergence not fully confirmed
- **Low confidence**: Limited evidence; search space may contain unexplored regions with better values

### Metrics

- **Best function value**: Primary metric — the maximum evaluation observed per function
- **Improvement rate**: Round-over-round improvement in best-known value
- **Ensemble calibration**: Leave-one-out prediction error monitored at each round

---

## 5. Assumptions and Limitations

### Assumptions

1. **Smoothness**: Functions are assumed to be reasonably smooth (continuous, without abrupt discontinuities), justifying local refinement
2. **Low multimodality**: Strategy assumes functions have a small number of optima; highly multimodal functions would be poorly served
3. **Stationarity**: Functions are fixed and do not change between rounds
4. **Bounded domain**: Global optimum lies within the provided [-5.0, 5.0] bounds

### Limitations

1. **Severe budget constraint**: ~104 queries per function cannot thoroughly explore a 10-dimensional space
2. **Linear ensemble aggregation**: Equal-weight averaging may not be optimal; performance-weighted aggregation could improve predictions
3. **Heuristic acquisition weights**: The four-component weights (0.45, 0.20, 0.20, 0.15) were set by judgment rather than optimised
4. **No transfer learning**: Each function is optimised independently; shared structure across functions is not exploited
5. **Scalability**: The approach does not scale to d > 15 without substantial modification (e.g., TuRBO-style local search)

---

## 6. Ethical Considerations

### Transparency and Reproducibility

- **Full documentation**: Every query, its reasoning, and its result are documented in the DATASHEET.md and query history CSV files
- **Code availability**: The complete pipeline is available in the public GitHub repository
- **Reproducibility caveat**: While the code is deterministic given a random seed, the human judgment that occasionally overrode acquisition function recommendations cannot be fully formalised

### Real-World Implications

- In production settings (hyperparameter tuning, clinical trial design), algorithmic methods with convergence guarantees should be preferred over this heuristic approach
- The principles demonstrated — exploration before exploitation, documenting decisions, acknowledging uncertainty — are directly transferable to any domain involving sequential decision-making under uncertainty

### Bias

- **Analytical bias**: The exploitation-heavy strategy in later rounds means the dataset (and therefore the surrogate models) over-represents promising regions and under-represents the full landscape
- **No social/demographic implications**: This project involves synthetic mathematical functions with no fairness concerns related to protected groups

---

## 7. Recommendations

Users of this approach should:

1. **Start with exploration**: Allocate at least 20% of the total query budget to space-filling designs before switching to model-based querying
2. **Monitor ensemble disagreement**: High disagreement signals regions where additional data would be most informative
3. **Perform sensitivity analysis**: Not all dimensions contribute equally; identify and focus on the influential ones
4. **Validate the surrogate**: Use leave-one-out cross-validation to check that the ensemble remains well-calibrated as data accumulates
5. **Adapt the strategy**: No single approach works for all functions — be prepared to adjust based on observed function characteristics

---

## References

- Mitchell, M., et al. (2019). Model Cards for Model Reporting. *Proceedings of FAT*.
- Gebru, T., et al. (2021). Datasheets for Datasets. *Communications of the ACM*.
- Shahriari, B., et al. (2016). Taking the human out of the loop. *Proceedings of the IEEE*.
- Wolpert, D. H. & Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Trans. Evolutionary Computation*.

---

*This model card is part of the BBO Capstone Project for the MIT Professional Certificate in Machine Learning and Artificial Intelligence.*
