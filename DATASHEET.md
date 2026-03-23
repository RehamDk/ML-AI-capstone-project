# Datasheet: BBO Capstone Project Query Data Set

## 1. Motivation

**Why was this data set created?**
This data set was created as part of the Black-Box Optimisation (BBO) capstone project to find the global optima (minimum or maximum) of eight unknown functions. The data set supports iterative optimisation by recording every query submitted and the corresponding function evaluation returned by the oracle.

**What task does it support?**
It supports the task of black-box function optimisation, where the learner has no access to the function's analytical form, gradient, or structure. The data set serves as the cumulative evidence base for making informed decisions about where to query next.

**Who created it and on whose behalf?**
The data set was created by the student (myself) as part of the MSc programme's required capstone component (Module 21), under the guidance of the course facilitators.

---

## 2. Composition

**What does the data set contain?**
The data set contains query–response pairs for eight unknown functions (F1–F8). Each record includes:

- **Function ID**: which of the eight functions was queried (F1 through F8).
- **Input vector**: the query point submitted, with each dimension specified to six decimal places.
- **Output value**: the scalar function evaluation returned by the BBO portal.
- **Round number**: which of the ten submission rounds the query belonged to (Round 1–10).

**What is the size?**
The data set contains 80 query–response pairs in total: 10 rounds × 8 functions × 1 query per function per round.

**What is the format?**
Data is stored in CSV format with columns: `function_id`, `round`, `x1`, `x2`, ..., `xn`, `f_value`. The number of input dimensions varies by function.

**Are there any gaps or missing values?**
There are no missing values. Every submitted query received a valid function evaluation from the portal. However, with only 10 data points per function, the coverage of the search space is inherently sparse.

---

## 3. Collection Process

**How were the queries generated?**
Queries were generated through a manual, iterative optimisation strategy that evolved across the ten rounds:

- **Rounds 1–3 (Exploration)**: Initial queries were distributed across the search space to build a broad understanding of each function's landscape. Strategies included grid-based sampling, random Latin hypercube sampling, and boundary probing.
- **Rounds 4–7 (Exploitation)**: Based on observed patterns, queries shifted towards regions that showed promising (extreme) function values. Interpolation between known good points and local refinement around the best-known values were the primary strategies.
- **Rounds 8–10 (Refinement)**: Final rounds focused on fine-tuning around suspected optima, testing nearby points at smaller step sizes to confirm convergence and improve precision.

**What strategy did you use?**
The strategy combined elements of exploration and exploitation, loosely following a Bayesian-inspired approach (without a formal surrogate model). Decisions were informed by manually examining trends, comparing function values across rounds, and hypothesising about function shape (e.g., unimodal vs. multimodal).

**Over what time frame?**
Data was collected over approximately 10 weeks, with one round of queries submitted per week.

---

## 4. Preprocessing and Uses

**Have you applied any transformations?**
- Function values were normalised per function (min–max scaling) for comparative visualisation across the eight functions.
- No transformations were applied to the raw input–output pairs used for decision-making; they remain in their original form.

**What are the intended uses?**
- Informing the next round of queries during the optimisation process.
- Analysing the convergence behaviour of the optimisation strategy.
- Supporting the reflection and reporting components of the capstone project.

**What are inappropriate uses?**
- This data set should not be used to train a general-purpose surrogate model, as it contains far too few observations (10 per function) for reliable function approximation.
- It should not be treated as a benchmark data set for comparing optimisation algorithms, since the query strategy was manual and non-reproducible in algorithmic terms.

---

## 5. Distribution and Maintenance

**Where is the data set available?**
The data set is available in the public GitHub repository for this capstone project (linked in the main README).

**What are the terms of use?**
The data set is shared for educational and peer-review purposes within the MSc programme. It may be referenced by peers for comparison but should not be redistributed outside the course context without permission.

**Who maintains it?**
The student (author of this project) maintains the data set. No further updates are planned after the completion of Round 10, as the capstone project's query phase is complete.

---

## 6. Additional Notes

**Limitations of the data set:**
- With only 10 observations per function, the data set provides a very sparse view of each function's landscape.
- Query placement was guided by human judgment, introducing subjective bias towards regions perceived as promising.
- The data set does not capture the reasoning behind each query, only the inputs and outputs. The accompanying reflections (in the discussion posts and this datasheet) provide that context.

**Ethical considerations:**
- This data set does not contain personal or sensitive information.
- Transparency is ensured by documenting the collection process, strategy, and limitations openly.
