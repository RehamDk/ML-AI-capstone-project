# Black-Box Optimization Capstone Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Round%203-yellow.svg)]()

## 👩‍💻 Author
**Reham Aldakhil**  
PhD Candidate, Imperial College London | Digital Health Leader | Clinical Informatics Expert

[![LinkedIn](https://img.shields.io/badge/LinkedIn-r--aldakhil-blue?logo=linkedin)](https://linkedin.com/in/r-aldakhil)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--6975--3858-green?logo=orcid)](https://orcid.org/0000-0002-6975-3858)

---

## 📋 Section 1: Project Overview

### What is Black-Box Optimization?

This capstone project tackles the challenge of optimizing eight unknown functions where the internal structure, gradients, and mathematical form are completely hidden. The only available information comes from querying specific points in the search space and observing their corresponding output values. This scenario mirrors real-world problems where we must make optimal decisions with limited information and expensive evaluations.

### Overall Goal and Real-World Relevance

The primary objective is to find the maximum value of each unknown function using as few queries as possible. This challenge is highly relevant in modern machine learning and optimization scenarios, including:

- **Hyperparameter Tuning**: Finding optimal ML model parameters where each training run requires significant computational resources
- **Drug Discovery**: Testing chemical compounds through costly laboratory experiments
- **Engineering Design**: Optimizing physical systems where each prototype test is expensive
- **Clinical Trials**: Determining optimal treatment protocols with limited patient trials
- **A/B Testing**: Making business decisions with constrained experiment budgets

The high-level approach involves developing a systematic, data-driven strategy that efficiently explores the search space while exploiting promising regions—all without knowing the underlying function structure.

### Career Relevance and Support

This project directly supports my career development in data science and healthcare analytics by:

1. **Building Practical Optimization Skills**: Many real-world healthcare problems involve optimizing systems that are expensive or difficult to evaluate (e.g., treatment protocols, resource allocation)

2. **Demonstrating Strategic Thinking**: Shows ability to make informed decisions under uncertainty with limited resources—critical in healthcare where experiments can affect patient outcomes

3. **Showcasing Technical Communication**: This GitHub repository demonstrates professional documentation of technical work for stakeholders, colleagues, and employers

4. **Developing Advanced ML Expertise**: Applies sophisticated machine learning techniques (SVMs, Gaussian Processes, Bayesian methods) to solve complex optimization problems

5. **Problem-Solving Portfolio**: Provides concrete evidence of applying theoretical knowledge to open-ended, challenging problems—valuable for data science roles in healthcare and beyond

---

## 📊 Section 2: Inputs and Outputs

### Input Format

Each query consists of a point in a multi-dimensional search space formatted as:

**Query Format**: `x_1=value,x_2=value,...,x_d=value`

**Constraints**:
- Each dimension `x_i` is bounded within the range: `-5.0 ≤ x_i ≤ 5.0`
- All values must be specified to exactly **6 decimal places**
- Dimensionality varies by function (typically 2-10 dimensions)
- Eight different unknown functions, each potentially with different dimensions

**Example Queries**:

*2-dimensional function:*
```
x_1=2.345678,x_2=-1.234567
```

*5-dimensional function:*
```
x_1=1.234567,x_2=-2.345678,x_3=0.123456,x_4=3.456789,x_5=-4.567890
```

### Output Format

The system returns a single numerical value representing the function's performance at the queried point:

**Response Format**: `y = f(x_1, x_2, ..., x_d)`

**Example Response**:
- Query: `x_1=2.345678,x_2=-1.234567`
- Response: `y = 15.234891`

This value represents the function output, with higher values indicating better performance (since the goal is maximization).

### Data Structure

As queries accumulate across rounds, historical data is structured as:

| x_1 | x_2 | x_3 | ... | x_d | y (output) |
|-----|-----|-----|-----|-----|------------|
| 2.345678 | -1.234567 | 0.123456 | ... | -0.987654 | 15.234891 |
| -0.987654 | 3.456789 | -2.345678 | ... | 1.234567 | 8.123456 |
| 1.111111 | -2.222222 | 3.333333 | ... | -4.444444 | 22.567890 |

This growing dataset informs machine learning models that guide subsequent query decisions, enabling increasingly intelligent exploration and exploitation strategies.

---

## 🎯 Section 3: Challenge Objectives

### Primary Goal

**Maximize** each of the eight unknown functions within the allocated query budget across multiple submission rounds.

### Key Constraints and Limitations

1. **Limited Query Budget**: Only a fixed number of function evaluations are available across all rounds, making each query valuable

2. **No Gradient Information**: Cannot compute derivatives or use gradient-based optimization methods—must rely on function values alone

3. **Unknown Function Structure**: No prior knowledge about whether functions are:
   - Convex or non-convex
   - Unimodal or multimodal (multiple peaks)
   - Smooth or discontinuous
   - Linear or highly non-linear

4. **High Dimensionality**: Some functions exist in 5-10 dimensional spaces, making exhaustive grid search computationally infeasible

5. **Response Delay**: Each query requires time to evaluate, emphasizing the need for efficient strategies

6. **No Function Identity**: Cannot distinguish which mathematical function is being optimized (Rastrigin, Ackley, Rosenbrock, etc.)

### Success Metrics

Performance is evaluated based on:
- **Best Value Found**: Maximum function value discovered across all queries for each function
- **Query Efficiency**: How quickly the approach converges to high-performing regions
- **Consistency**: Stable performance across all eight diverse functions
- **Improvement Rate**: Magnitude of improvement achieved in each query round

### Strategic Trade-offs

The challenge fundamentally requires balancing:
- **Exploration**: Sampling diverse regions to understand the overall landscape
- **Exploitation**: Focusing computational resources on known high-performing areas
- **Model Accuracy**: Building reliable surrogate models with limited training data
- **Computational Efficiency**: Algorithms that scale as the dataset grows

---

## 🛠️ Section 4: Technical Approach

### Round 1: Initial Exploration (Queries 1-8)

**Strategy**: Random sampling with space-filling design

**Implementation**:
- Generated uniformly distributed points across the entire search space [-5, 5]^d
- One query per function to establish baseline understanding
- No machine learning models employed—pure exploration phase

**Rationale**: With zero prior knowledge about function characteristics, unbiased random sampling provides initial information about different regions and establishes the value range for each function. This data serves as the foundation for model-based approaches in subsequent rounds.

**Key Learning**: Observed significant variation in function behaviors—some showed clear regional patterns while others appeared more chaotic, informing strategy adjustments for Round 2.

---

### Round 2: Gaussian Process Optimization (Queries 9-16)

**Strategy**: Gaussian Process Regression with Upper Confidence Bound (UCB) acquisition function

**Machine Learning Methods**:
- **Model**: Gaussian Process (GP) with RBF (Radial Basis Function) kernel
- **Acquisition Function**: Upper Confidence Bound (UCB)
  ```
  UCB(x) = μ(x) + β * σ(x)
  ```
  Where:
  - μ(x) = predicted mean (exploitation component)
  - σ(x) = predicted standard deviation (exploration component)
  - β = 2.0 (exploration parameter)

**Implementation Details**:
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Fit GP model to historical data
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp_model.fit(X_history, y_history)

# Generate candidates and compute UCB scores
predictions, std_devs = gp_model.predict(candidates, return_std=True)
ucb_scores = predictions + 2.0 * std_devs

# Select query with highest UCB score
next_query = candidates[np.argmax(ucb_scores)]
```

**Exploration vs Exploitation Balance**:
- **Exploitation (μ component)**: Targets regions with high predicted values
- **Exploration (σ component)**: Samples regions with high prediction uncertainty
- **Balance**: β = 2.0 provides roughly 50-50 balance between exploration and exploitation

**Rationale**: Gaussian Processes naturally quantify uncertainty in predictions, making them ideal for balancing exploration of uncertain regions with exploitation of promising areas. The UCB acquisition function provides a principled mathematical framework for this balance.

**Results**: Successfully identified promising regions for most functions, though some highly multimodal functions required more exploration than GP-UCB provided in just 8 additional queries.

---

### Round 3: SVM-Enhanced Optimization (Queries 17-24) [Current Round]

**Strategy**: Dual SVM approach combining classification and regression

**Machine Learning Methods**:

1. **Support Vector Classification (SVC)**:
   - **Purpose**: Divide search space into "high-performance" vs "low-performance" regions
   - **Threshold**: Set at 75th percentile of historical function values
   - **Kernel**: RBF kernel to capture non-linear decision boundaries
   - **Output**: Probability of a point being in a high-performance region

2. **Support Vector Regression (SVR)**:
   - **Purpose**: Predict actual function values within candidate regions
   - **Kernel**: RBF kernel for non-linear function approximation
   - **Output**: Point estimates of expected function values

3. **Combined Acquisition Function**:
   ```python
   score = α * predicted_value_norm + γ * prob_good_region + β * distance_norm
   ```
   
   Where:
   - **α = 0.5 (Exploitation)**: Weight for predicted function values from SVR
   - **γ = 0.3 (Classification Guidance)**: Weight for SVM classification confidence
   - **β = 0.2 (Exploration)**: Weight for distance to nearest known point
   
   All components normalized to [0, 1] range for fair weighting.

**Implementation Workflow**:
```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# 1. Prepare data: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_history)

# 2. Train SVM Classifier (high vs low performance)
threshold = np.percentile(y_history, 75)
y_binary = (y_history >= threshold).astype(int)
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_classifier.fit(X_scaled, y_binary)

# 3. Train SVR (value prediction)
svr_model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
svr_model.fit(X_scaled, y_history)

# 4. Generate and score candidates
candidates = generate_random_candidates(n=10000)
candidates_scaled = scaler.transform(candidates)

# Compute acquisition components
pred_values = svr_model.predict(candidates_scaled)
prob_good = svm_classifier.predict_proba(candidates_scaled)[:, 1]
distances = compute_min_distances(candidates_scaled, X_scaled)

# Normalize and combine
scores = 0.5 * normalize(pred_values) + \
         0.3 * prob_good + \
         0.2 * normalize(distances)

# Select best candidate
next_query = candidates[np.argmax(scores)]
```

**Hyperparameter Tuning**:
- Applied 5-fold cross-validation to select optimal C, gamma, and epsilon
- Prevents overfitting as dataset grows from 8→16→24 queries
- Used `GridSearchCV` for systematic hyperparameter search

**Unique Aspects of This Approach**:

1. **Region-Aware Search**: SVM classifier prevents wasting queries in regions definitively classified as low-performing, focusing resources on promising areas

2. **Dual Perspective**: Classification answers "where should I search?" while regression answers "what value should I expect?"—complementary information

3. **Adaptive Exploration**: Distance-based term ensures continued sampling of under-explored regions even when model is confident about certain areas

4. **Interpretability**: SVM decision boundaries can be visualized (in 2D/3D), providing insight into learned function structure

5. **Computational Efficiency**: SVMs scale well with growing datasets through kernel trick

**Why This Approach is Thoughtful**:

- **Progressive Refinement**: Each round systematically builds on previous learnings rather than starting from scratch
- **Multiple Information Sources**: Leverages classification confidence, value predictions, and spatial coverage simultaneously
- **Explicit Uncertainty**: Models what we know (predicted values) and what we don't (unsampled regions)
- **Defensible Decisions**: Can explain each query choice through quantitative model predictions and acquisition scores
- **Clinically Inspired**: As a healthcare researcher, I value interpretability—understanding *why* a decision is made, not just *what* decision to make

---

### Future Directions (Round 4+)

**Planned Enhancements**:

1. **Ensemble Methods**: Combine predictions from multiple models (GP, SVR, Random Forest, Gradient Boosting) to reduce individual model biases

2. **Local Optimization**: Once global promising regions are identified, switch to focused local search strategies (e.g., Nelder-Mead, Powell's method)

3. **Transfer Learning**: If patterns emerge across functions (e.g., similar optimal regions), leverage this shared structure to improve queries for remaining functions

4. **Adaptive Acquisition Weights**: Dynamically adjust α, β, γ based on:
   - Convergence rate (if improving rapidly → increase exploitation)
   - Prediction confidence (if uncertain → increase exploration)
   - Queries remaining (if running out → focus exploitation)

5. **Multi-Objective Optimization**: Balance multiple criteria simultaneously:
   - Maximize expected improvement
   - Minimize prediction uncertainty
   - Maximize distance from known points

**Bayesian Optimization Consideration**:

May implement full Bayesian Optimization framework:
- **Surrogate Model**: Gaussian Process for global function approximation
- **Acquisition**: Expected Improvement (EI) for mathematically principled exploration-exploitation
- **Sequential**: Update posterior beliefs after each query for optimal next selection

Expected Improvement formula:
```
EI(x) = E[max(0, f(x) - f_best)] = (μ(x) - f_best) * Φ(Z) + σ(x) * φ(Z)
```
Where Z = (μ(x) - f_best) / σ(x), and Φ, φ are standard normal CDF and PDF.

---

## 📁 Repository Structure

```
bbo-capstone-project/
├── README.md                          # This file
├── data/
│   ├── function_1_history.csv        # Historical queries for Function 1
│   ├── function_2_history.csv        # Historical queries for Function 2
│   ├── function_3_history.csv
│   ├── function_4_history.csv
│   ├── function_5_history.csv
│   ├── function_6_history.csv
│   ├── function_7_history.csv
│   └── function_8_history.csv
├── src/
│   ├── bbo_implementation.py         # Main optimization pipeline
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── base_optimizer.py         # Abstract base class
│   │   ├── random_optimizer.py       # Round 1 random sampling
│   │   ├── gp_optimizer.py           # Round 2 Gaussian Process
│   │   └── svm_optimizer.py          # Round 3 SVM-based optimizer
│   └── utils/
│       ├── __init__.py
│       ├── acquisition.py            # Acquisition functions
│       ├── visualization.py          # Plotting utilities
│       └── data_processing.py        # Data loading and formatting
├── notebooks/
│   ├── round1_random_exploration.ipynb      # Round 1 analysis
│   ├── round2_gp_optimization.ipynb         # Round 2 analysis
│   ├── round3_svm_strategy.ipynb            # Round 3 analysis
│   └── comparative_analysis.ipynb           # Cross-round comparison
├── results/
│   ├── queries/
│   │   ├── round1_queries.txt              # Round 1 submissions
│   │   ├── round2_queries.txt              # Round 2 submissions
│   │   └── round3_queries.txt              # Round 3 submissions
│   ├── figures/
│   │   ├── function_landscapes/            # 2D function visualizations
│   │   └── convergence_plots/              # Performance over rounds
│   └── models/
│       ├── round2_gp_models/               # Saved GP models
│       └── round3_svm_models/              # Saved SVM models
├── requirements.txt                         # Python dependencies
└── LICENSE
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.9
```

### Installation

```bash
git clone https://github.com/r-aldakhil/bbo-capstone-project.git
cd bbo-capstone-project
pip install -r requirements.txt
```

### Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Usage Example

```python
from src.optimizers.svm_optimizer import SVMOptimizer
import pandas as pd

# Load historical data for a function
data = pd.read_csv('data/function_1_history.csv')
X_history = data.iloc[:, :-1].values  # Query points
y_history = data.iloc[:, -1].values   # Function values

# Initialize optimizer
optimizer = SVMOptimizer(alpha=0.5, beta=0.2, threshold_percentile=75)

# Fit models to historical data
optimizer.fit(X_history, y_history, tune_hyperparameters=True)

# Generate next query suggestion
next_query, info = optimizer.suggest_next_query(
    X_history, 
    y_history, 
    n_candidates=10000
)

# Display results
print(f"Suggested next query: {next_query[0]}")
print(f"Predicted value: {info['predicted_values'][0]:.6f}")
print(f"Probability of good region: {info['prob_good_region'][0]:.3f}")
print(f"Acquisition score: {info['scores'][0]:.3f}")

# Format for submission
from src.utils.data_processing import format_query
formatted_query = format_query(next_query[0])
print(f"Formatted: {formatted_query}")
```

---

## 📈 Results Summary

| Round | Strategy | Best Values Found | Avg Improvement | Computational Cost |
|-------|----------|-------------------|----------------|-------------------|
| 1 | Random Sampling | Baseline | - | Low |
| 2 | GP + UCB | +15-30% | +22% | Medium |
| 3 | Dual SVM | +8-20% | +12% | Medium-High |

*Note: Values are illustrative—actual results updated after each round submission*

### Key Insights

**Round 1 → 2 Transition**:
- Largest performance jump due to shift from random to model-based approach
- GP effectively identified promising regions for most functions
- Some multimodal functions showed limited improvement, suggesting need for more exploration

**Round 2 → 3 Transition**:
- Incremental improvements as search space is better understood
- SVM classification helped avoid poor regions identified in earlier rounds
- Exploration component prevents premature convergence to local optima

**Cross-Function Patterns**:
- Lower-dimensional functions (2-3D) show faster convergence
- Higher-dimensional functions (8-10D) require more exploration budget
- No single strategy dominates all function types—adaptability is key

---

## 🔑 Key Takeaways

### Technical Lessons

1. **Sequential Learning is Powerful**: Each query provides information that compounds—late-round queries are informed by 16+ previous data points

2. **Uncertainty Quantification Matters**: Knowing what we don't know (via GP variance, SVM decision boundaries) is as valuable as predictions

3. **Balance is Dynamic**: Optimal exploration-exploitation balance shifts as data accumulates—early rounds need exploration, later rounds need exploitation

4. **Model Choice Depends on Data Size**: GPs excel with small data (Round 2), SVMs scale better as data grows (Round 3+)

5. **Interpretability Has Value**: Understanding why queries are chosen enables debugging, builds trust, and improves future strategies

### Real-World Parallels

This BBO challenge directly mirrors problems I encounter in healthcare research:

- **Clinical Trial Optimization**: Limited patient enrollment budget, need to find optimal treatment protocols
- **Resource Allocation**: Deciding where to invest healthcare resources with uncertain outcomes
- **Policy Evaluation**: Testing interventions with expensive, time-delayed feedback
- **Personalized Medicine**: Optimizing treatment for individual patients with limited trial-and-error opportunities

### Transferable Skills

1. **Strategic Decision-Making Under Uncertainty**: Making informed choices when complete information is unavailable
2. **Iterative Refinement**: Continuously improving approaches based on feedback
3. **Model Selection and Validation**: Choosing appropriate ML methods for problem characteristics
4. **Communication of Technical Work**: Explaining complex methods to diverse audiences
5. **Resource-Constrained Optimization**: Achieving goals within fixed budgets

---

## 📚 References

### Optimization Methods

1. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2015). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148-175.

2. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global optimization*, 13(4), 455-492.

3. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. *Advances in neural information processing systems*, 25.

### SVM Applications

4. Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and computing*, 14(3), 199-222.

5. Hsu, C. W., Chang, C. C., & Lin, C. J. (2003). A practical guide to support vector classification.

---

## 🏆 Acknowledgments

This project is part of the **MIT Applied Data Science Program** capstone requirements, focusing on practical application of machine learning optimization techniques in resource-constrained, high-uncertainty scenarios.

**Program**: Professional Certificate in Machine Learning and Artificial Intelligence  
**Institution**: Massachusetts Institute of Technology (MIT)  
**Focus**: Real-world application of advanced ML methods to open-ended optimization problems

Special thanks to the MIT teaching team for designing this challenging and practically relevant capstone project that bridges theoretical optimization concepts with real-world decision-making scenarios.

---

## 📧 Contact

**Reham Aldakhil**  
📧 r.aldakhil23@imperial.ac.uk  
🔗 [LinkedIn](https://linkedin.com/in/r-aldakhil) | [ORCID](https://orcid.org/0000-0002-6975-3858)

**Research Interests**: Healthcare Optimization, Clinical Decision Support, Digital Health, Machine Learning in Medicine

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔄 Project Status

- **Current Round**: 3 of 8
- **Queries Submitted**: 24 / [Total Budget]
- **Best Function Values**: [Updated after each round]
- **Next Milestone**: Round 4 submission implementing adaptive ensemble methods

---

*Last Updated: January 2026*

*This README is a living document and will be updated as the project progresses through subsequent query rounds.*
