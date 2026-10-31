# 📘 Lotto-Combo-Analyzer

> **A Scientific, Multi-Layer Analytical Framework for Lottery Combination Modeling and Experimental Prediction**
>
> *Version 1.1 — Scientific Edition*

## 🧩 Abstract

**Lotto-Combo-Analyzer** is a multi-layer analytical and machine-learning research framework designed to:

* Import and validate historical lottery draws.

* Characterize their mathematical, statistical, temporal, spectral, and probabilistic structure.

* Estimate number-level, pool-level, sum-level, and combination-level distributions.

* Train independent statistical, Machine Learning (ML), Deep Learning (DL), and probabilistic models.

* Fuse compatible model outputs through a calibrated ensemble.

* Generate, filter, score, and evaluate candidate combinations.

* Perform rigorous walk-forward backtesting against theoretical and empirical baselines.

> ⚠️ **Scientific Caution:** The system does **not** assume that lottery draws are predictable. All observed structure is treated as a research hypothesis, tested against:
>
> 1. Uniform theoretical baselines.
>
> 2. Rule-faithful random simulations.
>
> 3. Leakage-free out-of-sample evaluation.

## 📑 Table of Contents

 1. [Project Goals and Scientific Scope](#1-project-goals-and-scientific-scope)

 2. [Core Principles](#2-core-principles)

 3. [System Architecture](#3-system-architecture)

 4. [Lottery Data Model & Rules](#4-lottery-data-model--rules)

 5. [Mathematical Foundations](#5-mathematical-foundations)

    * [5.1 Information Theory](#51-information-theory)

    * [5.2 Divergence & Regime Detection](#52-divergence--regime-detection)

    * [5.3 Temporal Analysis](#53-temporal-analysis)

    * [5.4 Spectral Analysis](#54-spectral-analysis)

    * [5.5 Statistical Structure & Sum Distributions](#55-statistical-structure--sum-distributions)

    * [5.6 Correlation & Co-occurrence](#56-correlation--co-occurrence)

    * [5.7 Markov Models](#57-markov-models)

    * [5.8 Joint Probability Estimation](#58-joint-probability-estimation)

    * [5.9 Combination Space Modeling](#59-combination-space-modeling)

    * [5.10 Combination Entropy](#510-combination-entropy)

    * [5.11 Copula Modeling](#511-copula-modeling)

 6. [Feature Engineering](#6-feature-engineering)

 7. [Predictive Models](#7-predictive-models)

 8. [Uncertainty Quantification](#8-uncertainty-quantification)

 9. [Probability Calibration](#9-probability-calibration)

10. [Ensemble System](#10-ensemble-system)

11. [Candidate Generation & Scoring](#11-candidate-generation--scoring)

12. [Clustering & Association Analysis](#12-clustering--association-analysis)

13. [Simulation & Null Hypothesis Testing](#13-simulation--null-hypothesis-testing)

14. [Evaluation Metrics](#14-evaluation-metrics)

15. [Cross-Lottery Analytics](#15-cross-lottery-analytics)

16. [Data Acquisition & Storage](#16-data-acquisition--storage)

17. [API & Deployment](#17-api--deployment)

18. [Reproducibility](#18-reproducibility)

19. [Testing Strategy](#19-testing-strategy)

20. [Research Limitations](#20-research-limitations)

21. [Development Roadmap](#21-development-roadmap)

22. [Version](#22-version)

## 1. Project Goals and Scientific Scope

The system aims to:

* Build validated, versioned analytical datasets from official lottery sources.

* Quantify uncertainty, dependence, temporal structure, spectral properties, and regime changes.

* Train independent statistical, ML, DL, and probabilistic models.

* Generate and score candidate combinations under strict ruleset constraints.

* Evaluate all models using walk-forward chronological backtesting.

## 2. Core Principles

| Principle | Description | 
| ----- | ----- | 
| **Separation of Evidence vs. Hypothesis** | Clear distinction between empirical data and predictive assumptions. | 
| **Strict Lottery Isolation** | Data and rulesets for individual lotteries are kept strictly separated. | 
| **Explicit Model Declarations** | No black-box configurations; all hyperparameters and logic are versioned. | 
| **Reproducibility & Provenance** | Full lineage tracking for datasets, features, and model artifacts. | 
| **Scientific Caution** | Strict zero-overclaim policy regarding predictability. | 

## 3. System Architecture

```
Raw Data ──► Validation ──► Ruleset Resolution ──► SQLite Registry
                                                         │
                                                         ▼
                                                Feature Engineering
                                                         │
                                                         ▼
                                                 Scientific Layers
                                                         │
                                                         ▼
                                                  ML / DL Models
                                                         │
                                                         ▼
                                                    Calibration
                                                         │
                                                         ▼
                                                     Ensemble
                                                         │
                                                         ▼
                                                Candidate Generation
                                                         │
                                                         ▼
                                                      Scoring
                                                         │
                                                         ▼
                                                   Backtesting

```

## 4. Lottery Data Model & Rules

Each lottery entity maintains its own isolated configuration:

* **Ruleset History:** Historical changes in ball counts, bonus balls, or draw rules.

* **Number Pools:** Primary and secondary pool definitions.

* **Draw Structure:** Matrix size ($m$ out of $N$).

* **Feature Configuration & Models:** Isolated feature spaces and model weights.

* **Evaluation Results:** Historical out-of-sample backtest tracking.

> *Note: Ruleset changes are detected automatically but require human verification before registry entry.*

### Stage 4 feature engineering and stability evaluation

The Stage 4 engine, implemented in `src/feature_engineering.py`, computes:

- `S(f) = 1 - sigma(f_t) / mu(f_t)` for per-feature stability,
- `I(f) = (1 / M) * sum_m I_m(f)` for cross-model feature importance,
- chronological drift summaries across rolling windows,
- Monte Carlo stochastic null-hypothesis checks for feature drift under uniform Gaussian noise.

This keeps feature engineering strictly walk-forward, time-ordered, and benchmarked against random baselines rather than assuming that observed structure is predictive.

### Stage 5 model suite

The Stage 5 model suite, implemented in `src/model_suite.py`, includes:

- a uniform baseline and a frequency baseline using empirical counts,
- a Transformer-like deterministic encoder with softmax probabilities,
- an XGBoost-compatible proxy for model outputs when the library is unavailable,
- an MDN-style mixture model and a normalizing-flow proxy,
- a model ensemble that combines probability vectors with normalized non-negative weights,
- a Monte Carlo uniform benchmark for the null hypothesis.

All model outputs are validated to remain probability distributions, and the ensemble keeps weighting mathematically valid: Σ w_m = 1 and w_m >= 0.

### Stage 6 calibration and weight-constrained ensemble

The Stage 6 calibration engine, implemented in `src/calibration.py`, covers:

- temperature scaling: q_i = softmax(z_i / T),
- Platt scaling for binary calibration,
- isotonic-regression-style monotone calibration,
- probability-vector normalization after calibration,
- weighted ensemble fusion with normalized non-negative weights and valid sum-to-one constraints.

This ensures the ensemble maintains a valid probability distribution even after model output calibration.

### Confidence diagnostics and model release gates

`src/confidence_engine.py` evaluates chronological, out-of-sample categorical
probabilities with multiclass Brier score, log loss, expected calibration error,
and a paired bootstrap interval against the uniform null. Optional regime labels
produce separate diagnostics for each regime. The null comparison is evidence
against uniform probabilities on the supplied test set; it is not proof that
future lottery draws are predictable.

Feature drift can be checked with Jensen-Shannon divergence using
`compare_feature_distributions` in `src/feature_engineering.py`. The
`adaptive_ensemble_weights` helper in `src/model_suite.py` derives non-negative
weights from chronological validation losses, weighting recent windows more
heavily. These weights are only meaningful when the losses are genuinely
out-of-sample and use the same scoring protocol.

Candidate combinations can be selected as a diversified portfolio with
`select_diverse_candidates` in `src/candidate_engine.py`. It penalizes
Jaccard overlap between combinations; it does not claim to increase the chance
of a lottery win.

Model candidates are recorded in SQLite with their dataset SHA-256, artifact
URI, metrics, and lifecycle state. Promotion is atomic and requires all five
metrics to be explicitly `true`: `validation_passed`, `leakage_free`,
`calibration_passed`, `beats_null`, and `drift_within_tolerance`. Promotion
archives the previous active model for that lottery. This registry gate does
not train or persist model artifacts itself; a training pipeline must create
the artifact and report evidence before registration.

### CSV-triggered retraining

`POST /datasets/{lottery_id}/import` accepts a multipart CSV upload and
revalidates and retrains for each new content hash. The first upload is a
normal version-1 dataset snapshot: it does not need or compare against an older
CSV, and its response reports `first_dataset: true` and
`previous_active_model: null` if there is no active model. Re-uploading the
same filename and identical bytes is an idempotent no-op. A changed CSV under
the same filename creates the next dataset version and a new model candidate.

Before importing, register the lottery and a human-verified ruleset. A
single-pool ruleset uses inferred `numero N` or `nN` columns. Preview a file
with `POST /datasets/{lottery_id}/preview`; multiple-pool suggestions are
returned only when the pool bounds and draw counts identify exactly one valid
mapping. Otherwise the response sets `confirmation_required` and the mapping
must be reviewed.

Persist a confirmed mapping with
`PUT /lotteries/{lottery_id}/csv-schema-profiles/{ruleset_version}`. A profile
binds number-pool columns, optional date/event/session/time columns, explicitly
allowed attribute columns, ignored columns, and chronological-row-order
confirmation to one ruleset version. Retrieve it with `GET` on the same path.
When a profile exists, imports use it for omitted form fields and reject
missing or newly unclassified CSV columns so schema changes require review.
Allowed attributes and ignored columns are recorded for provenance; they are
not automatically used as predictive features.

Repeated dates require a unique event ID, session, or local draw-time identity,
and a configured schedule or local draw time to establish chronological order.
Date/session/time combinations and same-day scheduled timestamps must be
unique. A local draw-time column must use `HH:MM` or `HH:MM:SS`.

Configure the official schedule with
`PUT /lotteries/{lottery_id}/schedule`. Its JSON contract includes an IANA
timezone, effective date range, recurring weekday or anchored every-N-days
sessions, and explicit add/cancel exceptions. Use
`GET /lotteries/{lottery_id}/next-draw?after=<ISO-8601 timestamp>` to resolve
the upcoming event; the response includes its stable event ID, local scheduled
time, session, and timezone. Nonexistent daylight-saving local times are
rejected. The schedule identifies the event to predict; it does not imply
number-level predictive evidence.

`GET /lotteries/{lottery_id}/prediction` connects that event resolution to the
active model artifact. With no target parameters, it predicts for the next
scheduled event; a caller can instead provide the canonical scheduled
`event_id` or both `event_date` and `session_id`. The endpoint requires an
active model whose ruleset matches the target event and whose per-pool
marginal inclusion probabilities pass shape, bounds, and normalization
checks. It returns those marginal probabilities and bounded candidate lists.
For each fixed-size pool, candidate combinations are ranked by a normalized
conditional-Bernoulli subset distribution. Given per-number inclusion
propensities `p_i` and a draw size `k`, a subset `S` receives model-implied
mass proportional to `product(p_i / (1 - p_i))` for `i` in `S`, normalized
over all size-`k` subsets. The normalizer is computed in log space. This is an
explicit distributional assumption: conditioning the stored marginal
estimates this way does not guarantee that the resulting distribution's
marginals equal the stored estimates. The response exposes
`model_implied_combination_probability` separately from the marginal values;
it is not empirically calibrated confidence, a measured chance of winning, or
evidence of predictability. Candidate `score` remains a simple marginal-based
ranking diagnostic. The endpoint returns a conflict instead of substituting
a uniform or stale model when an active compatible artifact is unavailable.

Temporal weekday, month, quarter-of-year, and inter-draw-gap variables are evaluated with
expanding-window walk-forward Brier comparisons and Holm-Bonferroni correction.
They remain diagnostics and are not added to the frequency model. A calendar
effect is not treated as useful unless its out-of-sample evidence passes the
reported confidence and adjusted-significance checks. Each date must fall
within the selected ruleset range. Fully empty rows and records with no number
values are skipped and separately counted in the import response. Partially
populated draws, non-numeric, out-of-range, or duplicate-within-draw values
are rejected rather than silently dropped.

For example, after registering the lottery and ruleset:

```shell
curl.exe -X POST "http://localhost:8000/datasets/example-lottery/import" -F "file=@data/lotto_history.csv" -F "ruleset_version=v1" -F "chronological_order_confirmed=true"
```

Each accepted upload creates an immutable hash-addressed snapshot under
`data/datasets/`, a JSON frequency-inclusion model artifact under
`data/models/`, a dataset-version row, and a training-run record. The current
trainable implementation is a Laplace-smoothed frequency baseline. It uses
expanding-window walk-forward validation and compares its multi-label Brier
score to the uniform inclusion baseline. Promotion happens only if every
predefined validation, calibration, null-benchmark, and drift gate passes;
otherwise the new version remains a candidate and the previous active model
stays active. The first model is subject to the same gates and can remain a
candidate when evidence is insufficient. This pipeline does not claim that
lottery outcomes are predictable.

### Stage 1 domain and provenance registry

The Stage 1 contracts are implemented in `src/domain.py` using Pydantic v2, and
`src/registry.py` persists registry metadata in SQLite. Ruleset validity windows
are half-open date ranges (`effective_from <= draw_date < effective_to`); a
missing `effective_to` means the range has no end. Overlapping rulesets for the
same lottery are rejected, and each draw is bound to the ruleset version valid
on its draw date. Pools and draw numbers are stored separately so each pool is
validated against its own rules.

The SQLite registry is metadata and provenance storage, not a replacement for
the immutable raw CSV, Parquet, or JSONL source files. Register a source file
by its SHA-256 digest before storing draws that reference it. Initialize and
use a registry with:

```python
from src.domain import Lottery, NumberPool, PoolKind, Ruleset
from src.registry import SQLiteRegistry

registry = SQLiteRegistry("data/registry.sqlite3")
registry.initialize()
registry.register_lottery(
    Lottery(lottery_id="example-lottery", name="Example", jurisdiction="Example")
)
registry.register_ruleset(
    Ruleset(
        lottery_id="example-lottery",
        version="v1",
        effective_from="2020-01-01",
        pools=(
            NumberPool(
                pool_id="main",
                kind=PoolKind.PRIMARY,
                minimum=1,
                maximum=50,
                numbers_drawn=5,
            ),
        ),
    )
)
```

Ruleset registration never infers game rules from the sample CSV. Explicit,
human-verified pool definitions are required before importing draw records.

## 5. Mathematical Foundations

### 5.1 Information Theory

* **Shannon Entropy:**
  

  $$
  H(X) = -\sum p(x_i) \log p(x_i)
  $$

* **Conditional Entropy:**
  

  $$
  H(X\vert{}Y) = -\sum p(x, y) \log p(x\vert{}y)
  $$

* **Mutual Information:**
  

  $$
  I(X; Y) = \sum p(x, y) \log \left[ \frac{p(x, y)}{p(x)p(y)} \right]
  $$

* **Redundancy:**
  

  $$
  R = 1 - \frac{H(X)}{\log(n)}
  $$

### 5.2 Divergence & Regime Detection

* **Kullback–Leibler (KL) Divergence:**
  

  $$
  D_{\mathrm{KL}}(P \parallel Q) = \sum P(x) \log \left[ \frac{P(x)}{Q(x)} \right]
  $$

* **Jensen–Shannon Divergence:**
  

  $$
  D_{\mathrm{JS}}(P \parallel Q) = \frac{1}{2} D_{\mathrm{KL}}(P \parallel M) + \frac{1}{2} D_{\mathrm{KL}}(Q \parallel M)
  $$

* **Spectral Regime Transition Probability:**
  

  $$
  P(\text{regime\_change}) = \begin{cases} 1 & \text{if } D_{\mathrm{JS}}(p_{Wa}, p_{Wb}) > \tau \\ 0 & \text{otherwise} \end{cases}
  $$

### 5.3 Temporal Analysis

* **Temporal Density:**
  

  $$
  d_i(t) = \frac{1}{k} \sum x_i(j)
  $$

* **Return Probability:**
  

  $$
  P_{\text{return}}(i, t) = f(\Delta t_i, d_i(t), Z_t)
  $$

* **Collapse Probability:**
  

  $$
  P_{\text{collapse}}(i, t) = g(H_i(t), d_i(t), Z_t)
  $$

### 5.4 Spectral Analysis

* **Fourier Transform:**
  

  $$
  X_i(\omega) = \mathcal{F}\{x_i(t)\}
  $$

* **Spectral Entropy:**
  

  $$
  H_{\text{spectral}}(i) = -\sum p_i(\omega) \log p_i(\omega)
  $$

### 5.5 Statistical Structure & Sum Distributions

* **Combination Sum:**
  

  $$
  \Sigma(S) = \sum x_j
  $$

* **Exact Combinatorial Sum Distribution:**
  

  $$
  P(\Sigma = s) = \frac{\vert{}\{C \in \Omega : \sum C = s\}\vert{}}{\vert{}\Omega\vert{}}
  $$

* **Gaussian Approximation (Heuristic):**
  

  $$
  \Sigma \sim \mathcal{N}(\mu, \sigma^2)
  $$

### 5.6 Correlation & Co-occurrence

* **Pearson Correlation:**
  

  $$
  C_{ij} = \frac{\mathrm{cov}(X_i, X_j)}{\sigma_i \sigma_j}
  $$

* **Mutual-Information Dependence:**
  

  $$
  C_{\mathrm{MI}, ij} = I(X_i; X_j)
  $$

* **Co-occurrence Matrix:**
  

  $$
  M_{ij} = \sum \mathbf{1}(i \in D_t \text{ and } j \in D_t)
  $$

### 5.7 Markov Models

* **First-Order Transition:**
  

  $$
  T_{ij} = P(X_{t+1} = j \mid X_t = i)
  $$

* **Second-Order Transition:**
  

  $$
  T_{ijk} = P(X_{t+1} = k \mid X_t = j, X_{t-1} = i)
  $$

### 5.8 Joint Probability Estimation

$$
P(S) = P(X_1 = x_1, \dots, X_m = x_m)
$$

### 5.9 Combination Space Modeling

* **Combination Space Size:**
  

  $$
  \vert{}\Omega\vert{} = \binom{N}{m}
  $$

* **Conditional Distribution:**
  

  $$
  P_\theta(S = s \mid D, r), \quad s \in \Omega_r
  $$

### 5.10 Combination Entropy

* **Combination Entropy:**
  

  $$
  H(S) = -\sum_{s \in \Omega} P(S=s) \log P(S=s)
  $$

* **Normalized Combination Entropy:**
  

  $$
  H_{\mathrm{norm}}(S) = \frac{H(S)}{\log \vert{}\Omega\vert{}}
  $$

### 5.11 Copula Modeling

$$
F(x_1, \dots, x_d) = C(F_1(x_1), \dots, F_d(x_d))
$$

## 6. Feature Engineering

### 6.1 Feature Stability

$$
S(f) = 1 - \frac{\sigma(f_t)}{\mu(f_t)}
$$

### 6.2 Cross-Model Feature Importance

$$
I(f) = \frac{1}{M} \sum_{m=1}^M I_m(f)
$$

## 7. Predictive Models

* **Statistical Baselines:** Uniform random, historical frequency, lag-based models.

* **Transformer Encoder:** Sequence modeling for draw dynamics.

* **XGBoost:** Tabular gradient boosting over aggregated statistical features.

* **Normalizing Flows:** Generative modeling for density estimation in continuous representations.

* **Mixture Density Networks (MDN):** Multimodal output distribution estimation.

* **Sum Prediction Model:** Targeted regression for subset sum distribution bounds.

## 8. Uncertainty Quantification

* **Bootstrap Sampling:**
  

  $$
  \theta^* \sim \text{Bootstrap}(D)
  $$

* **Monte Carlo Dropout:**
  

  $$
  p(y \mid x) \approx \frac{1}{T} \sum_{t=1}^T f_{\text{dropout}}(x)
  $$

## 9. Probability Calibration

| Technique | Mathematical Formulation | 
| ----- | ----- | 
| **Platt Scaling** | $p_{\mathrm{cal}} = \sigma(a \cdot p + b)$ | 
| **Isotonic Regression** | $p_{\mathrm{cal}} = g(p)$ | 
| **Temperature Scaling** | $p_{\mathrm{cal}} = \mathrm{softmax}\left(\frac{z}{T}\right)$ | 

## 10. Ensemble System

### Validity Conditions

$$
\sum_{m=1}^M w_m = 1, \quad w_m \ge 0
$$

### Fusion Rule

$$
P(i) = \sum_{m=1}^M w_m P_m(i)
$$

## 11. Candidate Generation & Scoring

Candidate combinations $s$ are scored using a linear combination of normalized metrics:

$$
\mathrm{Score}(s) = w_1 P(s) + w_2 C(s) + w_3 T(s) + w_4 D(s) + w_5 G(s)
$$

Where:

* $P(s)$: Model probability score.

* $C(s)$: Coverage / spread metric.

* $T(s)$: Temporal coherence.

* $D(s)$: Distance / divergence score.

* $G(s)$: Structural group metric.

## 12. Clustering & Association Analysis

* **Unsupervised Clustering:** K-Means on feature vectors to detect draw profiles.

* **Association Rules Metrics:**
  

  $$
  \mathrm{Support}(A \Rightarrow B) = P(A \cap B)
  $$

  $$
  \mathrm{Confidence}(A \Rightarrow B) = P(B \mid A)
  $$

  $$
  \mathrm{Lift}(A \Rightarrow B) = \frac{P(A \cap B)}{P(A)P(B)}
  $$

## 13. Simulation & Null Hypothesis Testing

* **Null Hypothesis (**$H_0$**):** Draws are independent and uniformly distributed.

* **Test Statistic:** $T = f(\text{model outputs})$

* **Empirical p-value:**
  

  $$
  p = P(T^* \ge T)
  $$

## 14. Evaluation Metrics

* **Regression Metrics:** MAE, MSE, RMSE, $R^2$.

* **Probabilistic Metrics:** Log Loss, Brier Score.

* **Ranking & Accuracy:** Top-$k$ Accuracy, Match Hits ($6/6, 5/6, 4/6$).

* **Rule Quality:** Lift, Support, Confidence.

## 15. Cross-Lottery Analytics

Comparative framework evaluating model performance across multiple independent lottery systems using standardized, normalized metrics.

## 16. Data Acquisition & Storage

* **Immutable Raw Files:** Source data preserved without transformation.

* **SQLite Registry:** Relational storage for validated draws, rulesets, and metadata.

* **Provenance Tracking:** End-to-end data lineage records.

## 17. API & Deployment

* **Backend Engine:** FastAPI.

* **Containerization:** Docker & Docker Compose.

* **Asynchronous Execution:** Worker tasks and scheduled jobs for pipeline execution.

## 18. Reproducibility

* Versioned datasets.

* Versioned feature sets.

* Versioned model artifacts.

* Centralized experiment registry.

## 19. Testing Strategy

* **Scientific Tests:** Null hypothesis verification & statistical sanity checks.

* **ML Tests:** Data leakage validation & distribution drift detection.

* **API Tests:** Endpoint validation & contract testing.

* **Integration Tests:** End-to-end pipeline execution.

* **Regression Tests:** Performance baseline drift prevention.

## 20. Research Limitations

> 📌 **Important Disclaimer:** The framework does not claim predictability of pseudo-random or random lotteries. All observed deviations are evaluated against strict random baselines to prevent false discovery.

## 21. Development Roadmap

```
Phase 1 ──► Core Domain
Phase 2 ──► Data Import & Validation
Phase 3 ──► API & Docker Infrastructure
Phase 4 ──► Scientific Layers
Phase 5 ──► ML / DL Model Development
Phase 6 ──► Ensemble Integration
Phase 7 ──► Walk-Forward Backtesting
Phase 8 ──► Cross-Lottery Analytics
Phase 9 ──► Confidence Diagnostics & Model Promotion Governance

```

## 22. Version

**Version 1.1 — Scientific Edition**
