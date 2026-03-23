# Model Card: BBO Capstone Optimisation Approach

## 1. Overview

| Field | Details |
|-------|---------|
| **Name** | Iterative Manual Black-Box Optimisation (IMBBO) |
| **Type** | Heuristic, human-in-the-loop optimisation strategy |
| **Version** | 1.0 (Final, after 10 rounds) |
| **Author** | Student — MSc Programme, Module 21 Capstone |
| **Date** | 2026 |

---

## 2. Intended Use

**What tasks is this approach suitable for?**
- Low-budget black-box optimisation where the number of function evaluations is severely limited (e.g., 10 queries per function).
- Exploratory analysis of unknown function landscapes where the goal is to identify promising regions rather than guarantee global optimality.
- Educational settings where the emphasis is on understanding optimisation principles, transparency, and iterative reasoning.

**What use cases should be avoided?**
- High-dimensional optimisation problems (beyond ~5 dimensions), where manual reasoning about the search space becomes intractable.
- Safety-critical applications where convergence guarantees or probabilistic bounds on optimality are required.
- Automated pipelines — this approach requires human judgment at every step and cannot be deployed as an algorithm.

---

## 3. Strategy Details

The optimisation approach evolved across three phases over the ten rounds:

### Phase 1: Exploration (Rounds 1–3)
The initial rounds prioritised broad coverage of the search space. Techniques included:
- **Boundary and midpoint sampling**: querying the corners, edges, and centre of the feasible region to establish baseline function values.
- **Spread-out random sampling**: distributing points to avoid clustering and maximise spatial coverage.
- **Dimensionality awareness**: for higher-dimensional functions, focusing on axis-aligned slices to reduce the effective search complexity.

### Phase 2: Exploitation (Rounds 4–7)
With initial data in hand, the strategy shifted towards regions showing the most promising function values:
- **Local refinement**: querying points near the current best-known value, stepping in each dimension individually to estimate local gradients.
- **Interpolation**: placing queries between two good points to check for better intermediate values.
- **Pattern recognition**: identifying whether functions appeared unimodal, multimodal, or symmetric, and adjusting the search accordingly.

### Phase 3: Convergence (Rounds 8–10)
The final rounds aimed to confirm suspected optima and improve precision:
- **Fine-grained steps**: reducing step sizes to six-decimal-place precision around the best-known point.
- **Cross-validation**: re-querying slightly perturbed versions of the best point to verify stability.
- **Filling gaps**: allocating one or two final queries to previously unexplored regions as a hedge against missed global optima.

### How the approach evolved
The balance between exploration and exploitation shifted from roughly 80/20 in early rounds to 20/80 in later rounds. Decisions were informed by manually plotting function values against input dimensions and looking for trends, peaks, and valleys.

---

## 4. Performance

### Results Summary

Performance is reported per function as the best (minimum or maximum, depending on the objective) function value found across the 10 rounds.

| Function | Dimensionality | Best Value Found | Round Achieved | Confidence |
|----------|---------------|-----------------|----------------|------------|
| F1 | — | (to be filled) | — | Medium |
| F2 | — | (to be filled) | — | Medium |
| F3 | — | (to be filled) | — | Medium |
| F4 | — | (to be filled) | — | Low |
| F5 | — | (to be filled) | — | Medium |
| F6 | — | (to be filled) | — | Low |
| F7 | — | (to be filled) | — | Medium |
| F8 | — | (to be filled) | — | Low |

*(Fill in the table above with your actual results from the portal.)*

**Confidence levels** reflect subjective assessment:
- **High**: the best value was confirmed by nearby queries showing worse values in all directions (strong evidence of local optimum).
- **Medium**: the trend suggests proximity to an optimum, but convergence was not fully confirmed.
- **Low**: limited evidence; the search space may contain unexplored regions with better values.

### Metrics used
- **Best function value**: the primary metric — the best (most extreme) evaluation observed.
- **Improvement rate**: how much the best-known value improved from round to round, used to assess convergence.
- **Spatial coverage**: qualitative assessment of how well the queries covered the feasible region.

---

## 5. Assumptions and Limitations

### Key Assumptions

1. **Smoothness**: the functions are assumed to be reasonably smooth (continuous, without abrupt discontinuities), so that nearby inputs produce similar outputs. This justified local refinement strategies.
2. **Low multimodality**: the strategy assumed functions have a small number of optima. Highly multimodal functions with many local optima would be poorly served by this approach.
3. **Stationarity**: the functions are assumed to be fixed (not changing between rounds), so that earlier observations remain valid throughout.
4. **Bounded search space**: the feasible region is assumed to be the domain provided by the portal, and the global optimum is assumed to lie within it.

### Limitations

1. **Severe budget constraint**: with only 10 queries per function, the approach cannot explore the search space thoroughly. Many regions remain unsampled.
2. **No surrogate model**: unlike Bayesian optimisation, this approach does not build a probabilistic model of the function, so it cannot quantify uncertainty or make statistically optimal query decisions.
3. **Human bias**: query placement was guided by human intuition, which introduces biases — such as a tendency to focus on regions that "look promising" based on a small number of observations, potentially ignoring other areas.
4. **Scalability**: the manual approach does not scale to high dimensions or large query budgets. Algorithmic methods would be necessary for more complex problems.
5. **No convergence guarantee**: there is no mathematical assurance that the best value found is near the true global optimum.

---

## 6. Ethical Considerations

### Transparency and Reproducibility
- **Transparency**: every query and its result are documented in the accompanying datasheet (DATASHEET.md). The reasoning behind each round's strategy is described in the discussion board reflections and in this model card.
- **Reproducibility**: while the exact sequence of queries could be replicated, the human judgment that guided them cannot be fully formalised. Another researcher following the same high-level strategy would likely produce different query sequences. This is a fundamental limitation of human-in-the-loop approaches.
- **Peer review**: by publishing the datasheet, model card, and query history in a public GitHub repository, the work is open to scrutiny and feedback, supporting academic integrity.

### Real-World Adaptation
- In real-world applications (e.g., hyperparameter tuning, engineering design), the principles demonstrated here — exploration before exploitation, documenting decisions, acknowledging uncertainty — are directly transferable.
- However, production systems should use algorithmic optimisation methods (e.g., Bayesian optimisation, evolutionary strategies) that provide uncertainty estimates and scale to larger budgets and dimensions.

### Bias and Fairness
- This project involves mathematical functions with no social or demographic implications. There are no fairness or bias concerns related to protected groups.
- The primary bias concern is analytical: human-guided search may systematically overlook certain regions of the function landscape.

---

## 7. References

- Gebru, T., et al. (2021). Datasheets for Datasets. *Communications of the ACM*.
- Mitchell, M., et al. (2019). Model Cards for Model Reporting. *Proceedings of FAT*.
- Course materials from Mini-lessons 21.1 and 21.2.
