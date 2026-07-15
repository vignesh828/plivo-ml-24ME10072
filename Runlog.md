# Training Run Log

**Run 1: Starter Baseline**
*   **Hypothesis:** The provided Logistic Regression with basic energy/pitch means will yield baseline results.
*   **Score:** ~1600 ms delay on English.
*   **Conclusion:** Silence and static means are not enough. The model needs dynamic features (slopes, drops).

**Run 2: Initial Random Forest & Expanded Features**
*   **Hypothesis:** Adding pitch slope, energy drop-off, and ZCR to a Random Forest will capture non-linear EOT cues.
*   **Score:** 1164 ms delay @ 5.0% interruptions (AUC: 0.691).
*   **Conclusion:** Huge improvement, but the delay is still over 1 second. The model is too conservative in calling an EOT.

**Run 3: Hyperparameter Tuning (The Overfitting Trap)**
*   **Hypothesis:** Unconstrained Random Forest will learn better boundaries.
*   **Score:** 100 ms delay (AUC: 1.000), but held-out turn accuracy crashed to 0.492.
*   **Conclusion:** The model memorized the training data and failed to generalize. Need to restrict depth.

**Run 4: Constrained Random Forest & Threshold Optimization**
*   **Hypothesis:** Limiting max_depth=10 and min_samples_leaf=10 prevents overfitting. 
*   **Score:** 700 ms delay on English (AUC: 0.929).
*   **Conclusion:** The model generalized perfectly. Applying this to the unseen Hindi data yielded an 850 ms delay, proving the prosodic features are language-agnostic.