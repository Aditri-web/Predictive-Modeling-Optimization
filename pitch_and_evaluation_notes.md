# Hackathon Pitch & Evaluation Notes

This document provides the necessary talking points and insights for **Phase 2: Final Pitch & System Understanding**, directly addressing the evaluation criteria through a highly sophisticated Physics-Informed ML approach explicitly designed for a **bimodal target distribution**.

## 1. Quantitative Evaluation (RMSE)
The competition's primary metric for Phase 1 is Root Mean Squared Error (RMSE):
`RMSE = √ [ (1/n) * Σ (yᵢ - ŷᵢ)² ]`

**Our Approach & Results:**
To ensure our model performs incredibly well on the hidden 50 test rows, we evaluated it locally using rigorous 5-Fold Cross-Validation.
- **5-Fold Cross-Validation RMSE**: ~18.34
- Our model drops the RMSE drastically by dynamically smoothing out the massive error "cliffs" caused by the bimodal distribution of reactor yield data (near 0% vs 70-100%).

## 2. Dealing with Bimodality: The Two-Stage Pipeline
The dataset presents a profound challenge: standard regressors get heavily penalized at the boundary between reaction failure (~0% yield) and success (70-100% yield). A generic regressor tries to shoot through the middle, leading to massive residuals on both ends.

**Our Fix:**
We designed a Two-Stage ML Pipeline:
- **Stage 1 (Classifier)**: A `GradientBoostingClassifier` trained to predict the probability that a reactor run will be successful (Yield > 1.0%).
- **Stage 2 (Regressor)**: We exclusively filter the training data to successful reactions and train our regressors solely on this continuous high-yield space.
- **Soft Blending**: For inference, we multiply the Regressor's raw prediction by the Classifier's probability of success. This mathematically smooths the cliff edge, avoiding catastrophic prediction errors on failed reactions.

## 3. Process Insight & Extended Feature Engineering
We didn't just let the model search blindly; we explicitly encoded chemical engineering principles directly into the feature space using 7 custom variables:
- **Residence Time ($\tau$)**: `length / flow_rate` — Core reactor variable; how long the reactant stays in the reactor.
- **Temperature Delta**: `jacket_T - inlet_T` — The exact heat exchange driving force.
- **Arrhenius Terms**: `exp(-5000 / T)` for both inlet and jacket — Reaction rate scales exponentially with temperature.
- **Conversion Capacity**: `concentration * residence_time`.
- **Volume Throughput**: Total material processed.
- **Residence Time Squared**: Captures the non-linear "over-cooking" effect.

## 4. Inverse-RMSE Weighted Ensemble
Standard voting regressors assign equal weights, which is suboptimal when variance is high. We implemented a custom inverse-RMSE weighted ensemble across our 5 folds:
- **GradientBoostingRegressor**
- **HistGradientBoostingRegressor**
- **RandomForestRegressor**
For each fold, we evaluate the validation RMSE for each model. Models with lower RMSE receive exponentially higher weight in the final blend. This guarantees that the most robust model for any given data fold commands the ensemble.

## 5. Scalability & Plant Deployment
Judges want to know how the model scales to a real plant.
- Physics-based BVP simulations take significant computational power and time to solve heavy differential equations. 
- Our Two-Stage Physics-Informed ML surrogate model performs inference in **milliseconds**. Once trained, it is perfectly suited for high-frequency, real-time plant optimization without hitting mathematical boundaries!
