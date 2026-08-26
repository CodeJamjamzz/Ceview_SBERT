# Machine Learning Experimentation Methodology

This document serves as a detailed record of the specific engineering practices, workflows, and optimization techniques we employed while training the Ceview SBERT multi-class classifier.

## 1. The Experimentation Workflow (Avoiding Data Leakage)
To ensure our model is scientifically robust and will perform well in the real world, we strictly follow the standard ML engineering workflow:

1. **The "Wide Net" Phase**: We run multiple experiments with different configurations (changing optimizers, learning rates, layer architectures). During this phase, we **only** evaluate the model against the 10% Validation Set. 
2. **Selecting the Champion**: We compare the experiments in Weights & Biases (W&B) and select the single "Champion Model" based entirely on the lowest `val_loss` and highest `val_f1_macro`. 
3. **The Final Test**: Only after the Champion Model is selected do we evaluate it against the 10% Test Set. We calculate the `test_loss` and test metrics **exactly once**. By never checking the Test Set during experimentation, we completely avoid "Data Leakage" (accidentally training the model to memorize the test data).

## 2. Training Optimization Techniques
During our experimentation, we explored and implemented several advanced techniques to squeeze the best performance out of the model:

* **Model Checkpointing (Callback)**: We do not simply save the model at the final epoch. We implemented a callback mechanism that actively tracks `val_loss` during training. If the model achieves a new lowest `val_loss`, it instantly saves those specific weights (`best_model.pth`). If the model starts to overfit in later epochs, we still retain the absolute best version of the brain.
* **Learning Rate Schedulers (`ReduceLROnPlateau`)**: We explored using a scheduler to dynamically adjust the learning rate. If the `val_loss` stops improving (plateaus) for 2 epochs, the scheduler cuts the learning rate in half. This prevents the model from taking steps that are too large and bouncing out of the optimal minimum.
* **Advanced Optimizers**: We experimented with swapping the standard `Adam` optimizer for `AdamW` (Adam with decoupled Weight Decay) to provide better regularization and prevent the model from overfitting on our small dataset.

## 3. Reproducibility (Random Seeds)
Deep learning is highly non-deterministic due to random weight initialization, data shuffling, and Dropout layers. To ensure that our experiments are perfectly comparable:
* We implemented a `set_seed(42)` function that locks PyTorch, NumPy, and Python's random number generators.
* This makes every run 100% deterministic. If a new configuration gets a better `val_loss`, we can confidently attribute it to the configuration changes rather than "lucky" random initialization, eliminating the need to train a single configuration multiple times to find an average.

## 4. Current Architecture Status (SBERT Frozen)
As of our latest finalized attempt:
* The SBERT backbone (`paraphrase-multilingual-MiniLM-L12-v2`) is strictly **frozen** (`requires_grad = False`). It acts purely as a static feature extractor.
* We have **not** yet attempted an "unfrozen" training run where the SBERT layers are allowed to learn and adjust to our specific tourism dataset.
* *(Note: The label `sbert-unfrozen-final` is currently used as a naming convention placeholder in our integration docs, but the architecture itself remains frozen pending future experiments).*

## 5. Evaluation Metric Bug Fix
During our testing phase, we discovered and resolved a major calculation bug:
* The model's final layer is a `Sigmoid`, meaning it outputs raw probabilities (0 to 1).
* The evaluation script was accidentally using `nn.BCEWithLogitsLoss()`, which applies a second Sigmoid layer on top of the probabilities. This mathematically skewed the loss calculation, resulting in an artificially massive `test_loss` (e.g., `0.62`) despite having excellent Accuracy (78%) and F1 Scores (92%).
* **The Fix**: We corrected the evaluation script to use `nn.BCELoss()`, which correctly expects probability inputs, bringing the `test_loss` down to match the `val_loss`.
