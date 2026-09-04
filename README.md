# Entropy-GameTheory-EV-Purchases

# EntropyGameTheoryEngine 🏎⚡

An advanced Python-based framework leveraging **Information Theory, Game Theory ensembles, and Psychological Segmentation** to solve complex tabular classification problems. 

This engine is specifically adapted and optimized for the **Kaggle Playground Series (Season 6, Episode 9)** competition: *Predicting Electric Vehicle Purchases*.

## 🚀 Architectural Core & Philosophy

Standard Gradient Boosting pipelines fail on synthetic datasets (like CTGAN outputs) due to logical inversions and generative anomalies. **EntropyGameTheoryEngine** bypasses these limitations using a multi-tiered approach:

1. **Autonomous Data Auditor:** Calculates individual row entropy against an Out-of-Fold (OOF) triad matrix. Instead of blindly filtering data, it isolates generative noise (e.g., CTGAN artifacts violating Simpson's paradox) and injects it back as a structured meta-signal (`gen_error_signal`).
2. **Psychological & Cohort Segmentation:** Translates raw consumer behavior metrics into non-linear behavioral archetypes:
   * **Loyal Conservative Index:** Captures the structural inertia of experienced vehicle owners resistant to switching proven combustion cars.
   * **Early Adopter Trendsetter:** Pinpoints high-income, tech-forward demographics eager for ecological integration.
   * **Economic Inertia Coeff:** Models consumer resistance when high income meets minimal daily commute utility.
3. **The Monolithic Triad:** An ensemble of three state-of-the-art architectures (`LightGBM` + `XGBoost` + `CatBoost`) accelerated via hardware GPU/CUDA kernels.
4. **Nelder-Mead Rank Fusion:** Replaces standard mean blending with non-parametric Spearman Space optimization, preventing private leaderboard shake-ups by utilizing automated weight-matching.

## 📊 Performance Benchmark

| Pipeline Stage | Local OOF AUC | Kaggle Public Score | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline Triad** | `0.94165` | `0.93935` | Standard tabular features with equal blending |
| **Sterile Cleansing (Plan A)** | `0.96977` | `0.93910` | Overfitting due to hard noise filtering |
| **Entropy Adaptive (v6.1)** | **`0.94184`** | **`0.94114+`** | Solid macro-validation using structural noise flags |

## 🛠 Project Structure

```text
├── EntropyGameTheoryEngine.py  # Core execution monolith for Kaggle environment
├── README.md                   # Documentation and architectural manual
└── requirements.txt            # Python environment constraints
```

## ⚙️ Fast GPU Deployment

### Prerequisites
Ensure your environment (Kaggle Notebooks or Google Colab) is running a CUDA-compatible GPU instance (T4, P100, or A100).

### Installation
```bash
pip install -r requirements.txt
```
*Dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `scipy`.*

### Execution
Run the monolith script inside your workspace. The engine automatically handles data imputation, factor synthesis, GPU initialization, early stopping calibration, and generates the final submission file.

```python
# Execution command within notebook environment
!python EntropyGameTheoryEngine.py
```

## 🧠 Key Insights Uncovered by the Auditor

The engine successfully diagnosed critical anomalies injected by the synthetic data generator:
* **Infrastructure Paradox:** The generator frequently assigned a high conversion rate (`Will_Buy_EV = 1`) to zero-ecological-concern profiles lacking home charging access, confusing native tree-based learners.
* **Simpson's Paradox Collision:** A hidden correlation loop between `Home_Charging_Possible` and external station density was successfully decoupled using our explicit categorical mapping matrix.

## 📝 License
This framework is distributed under the **MIT License**. Feel free to adapt and scale the Monolithic Triad architecture for other Kaggle Playground tournaments!
