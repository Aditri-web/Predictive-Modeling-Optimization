# Hackathon Pitch & Evaluation Notes

## 1. Quantitative Evaluation (RMSE)
`RMSE = √ [ (1/n) * Σ (yᵢ - ŷᵢ)² ]`

| Version | Overall Combined 150-Sample RMSE | Core Upgrades & Notes |
|---|---|---|
| Baseline (V1) | ~21.78 | Vanilla GBR, no feature engineering or zero-handling |
| V2 | ~18.02 | Two-Stage Classifier + Regressor, basic residence time |
| V3 | ~14.07 | Stacking + Optuna + positive Ridge + per-fold threshold |
| V10 | ~13.36 | Multi-Ea Kinetics + Regressor SVR + Feature Selection |
| V14 | ~12.94 | Logit Target Transform + Distance KNN |
| V17 | ~12.75 | Class-Weighted Classifier + Neural MLP + GP |
| **V20 (Current Best)** | **`12.5616`** | **Feature Selection Optimization + Multi-Seed Stacking + Neural MLP + GP** |

Our final V20 model achieves a **~42.3% overall RMSE reduction** from baseline, reaching the lowest overall combined out-of-fold RMSE of **`12.5616`** across all 150 training samples.

---

## 2. Core Insight: Bimodal Distribution & Soft-Gating
The dataset presents a profound challenge — nearly 31% of training reactions are near-zero yield (failures), while the remaining 69% are high-yield (60–100%). A standard regressor trained on both groups simultaneously anchors predictions near the wrong global mean (~36%).

**Our Fix — Two-Stage Pipeline with Sigmoid Soft-Gating:**
- **Stage 1 (Soft-Voting Classifier Ensemble)**: Combines XGBoost, LightGBM, CatBoost, ExtraTrees, Random Forest, and SVC to achieve **89.3%+ accuracy** in separating zero-yield reactions from successful ones.
- **Stage 2 (Stacking Regressor Ensemble)**: Trained exclusively on non-zero yield reactions ($N=104$).
- **Sigmoid Soft-Gating**: Replaces hard binary step functions with a continuous sigmoid gate:
  $$g(P) = \frac{1}{1 + \exp(-15 \cdot (P - \theta))}$$
  This eliminates cliff boundary penalties where borderline classification probabilities previously caused sharp prediction jumps.

---

## 3. Advanced Physics & Reaction Kinetics Feature Engineering (54 Vectors)
We explicitly encoded chemical kinetics and thermodynamics into the feature space:
- **Multi-Activation Energy Arrhenius Terms**: $\exp(-E_a / (RT))$ across $E_a/R \in [2.5\text{k}, 3.2\text{k}, 4.5\text{k}, 6\text{k}, 8\text{k}, 10\text{k}]$ to model both primary synthesis ($A \rightarrow B$) and competing side/degradation ($B \rightarrow C$) rate kinetics.
- **Analytical Kinetic Rate Model**: $Y_{\text{kinetic}} = \frac{k_1}{k_2 - k_1} (\exp(-k_1 \tau) - \exp(-k_2 \tau)) \cdot C_{\text{inlet}}$.
- **Damköhler Numbers**: $Da = k(T) \cdot \tau$ — ratio of reaction rate to mass transfer rate.
- **Residence Time Dynamics**: $\tau = \text{length} / \text{flow\_rate}$, $\tau^2$, and $\ln(1 + \tau)$ — over-cooking non-linear penalties.
- **Thermal Profiles**: Temperature ratio ($T_{\text{jacket}}/T_{\text{inlet}}$), relative drive $(T_{\text{jacket}} - T_{\text{inlet}})/T_{\text{inlet}}$, squared delta, and mean reaction temperature.

---

## 4. Logit Target Transformation for Bounded Yields
In non-zero yield reactions ($y \in (0, 100]$), standard MSE loss pulls predictions towards the sample mean (~60%), causing regression attenuation on high (95%+) and low (15%) yields.

**Our Fix:**
- Transform targets into un-bounded logit space: $y_{\text{logit}} = \ln\left(\frac{y/100}{1 - y/100}\right)$.
- Base regressors predict in logit space, and final predictions are inverted via $\hat{y} = 100.0 \times \sigma(\hat{y}_{\text{logit}})$.
- **Result**: Dropped out-of-fold errors by accurately capturing extreme yield bounds.

---

## 5. Non-Tree Kernel Models & Non-Negative Ridge Stacking
- **Smooth Kernel & Neural Models**: Integrated distance-weighted KNN ($K=5$), Support Vector Regressor (SVR with RBF kernel), Neural MLP (`MLPRegressor` with `tanh` activation), and Gaussian Process Regressor (Matern kernel) alongside tree models (XGBoost, LightGBM, CatBoost, ExtraTrees, Random Forest, GBR).
- **Non-Negative Meta-Learner**: `Ridge(positive=True, alpha=3.0)` trained on Out-Of-Fold predictions enforces positive weights—preventing models from subtracting value from the ensemble.
- **Multi-Seed Averaging**: Ensembled predictions across 5 independent random seeds ($[42, 100, 2024, 777, 999]$) to minimize variance.

---

## 6. Scalability & Real-Time Plant Deployment
Physics-based boundary value problem (BVP) simulations are computationally prohibitive in real-time plant control loops.
- Our surrogate performs inference in **milliseconds**, making it suitable for high-frequency real-time optimization.
- The physics-informed features ensure the model extrapolates sensibly under changing plant setpoints without black-box failure modes.
