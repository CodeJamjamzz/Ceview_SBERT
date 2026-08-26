# Training Workflow

## Environment
Training is performed on **Google Colab** using **PyTorch**. This IDE (local repo) is in charge of code changes, but the heavy lifting of training is executed on Colab.

## Pipeline Diagram
```mermaid
graph LR
  A[Local Code Changes] -->|Commit & Push| B(GitHub)
  B --> C[Pull to Google Colab]
  C -->|Run Training Notebook| D{Model Training}
  D -->|Evaluate Metrics| E[Export Model & Results]
  E -->|Push Results (JSON ONLY) Back| B
  E -.->|Upload weights manually| F(HuggingFace / S3)
  B -->|Pull to Local| A
```

> [!WARNING]
> Do **NOT** push the `saved_models/` directory or any `.pth` files back to GitHub. They exceed GitHub's 100MB file limit. Always push the model weights to an external storage service (like HuggingFace) and just push the updated `experiments/latest.json` file back to GitHub.

## Training Execution

**Google Colab Setup:**
1. Clone this repository into the Colab environment.
2. Ensure you have the `data/` folder populated with your JSON datasets.
3. Install dependencies: `pip install -r requirements.txt` (or manually install `torch`, `transformers`, `scikit-learn`, `wandb`).

### Data Preprocessing
Before the training loop begins, `model/preprocessing.py` will:
1. Scan the `data/` folder for all `.json` files. The expected schema for each object in the JSON array is:
   ```json
   {
     "id": 1,
     "description": "Text...",
     "uvp": "Text...",
     "services": ["Service A", "Service B"],
     "labels": ["Coastal & Island", "Adventure & Nature"]
   }
   ```
2. Convert the `services` array into a single comma-separated string.
3. Pass the strings through the `transformers.AutoTokenizer` using the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` tokenizer.
4. Multi-hot encode the `labels` array into a 1x8 float tensor matching the explicit index map defined in `docs/model.md`.

After preprocessing, the `TourismDataset` serves these tokenized inputs and labels to the training loop.

### Dataset Preparation and Class Imbalance
The dataset contains approximately 1,383 examples with the following class distribution:
- Accommodation: 423
- Culinary: 346
- Adventure: 298
- Urban: 285
- Coastal: 282
- Cultural: 274
- OUT_OF_SCOPE: 226
- Theme Parks: 155

This reveals an imbalance ratio of ~2.73 (423 ÷ 155), which constitutes a **mild/moderate imbalance** (ratio between 2:1 and 5:1). Because the imbalance is not severe, it is not necessary to artificially duplicate (oversample) or remove (undersample) data to make classes strictly equal.

Instead, the recommended and implemented approach is to perform a **stratified 80/10/10 split (Train/Validation/Test)**. 
- **Train (80% / ~1,106 examples):** Used to actually teach the model.
- **Validation (10% / ~138 examples):** Used during training to monitor progress and tune hyperparameters.
- **Test (10% / ~138 examples):** A completely held-out set used only at the very end to provide an unbiased evaluation of the model's performance.

A stratified split ensures that each class maintains roughly the same proportion across all three sets, ensuring that even the smallest classes have representation in the Test set.

## Reproducibility (Random Seeds)
Deep learning inherently relies on randomness (e.g., initial weight allocation, shuffling data, dropout). If left unchecked, training the exact same architecture twice will yield different results due to this random variance, making it impossible to confidently compare two different models.
To fix this, a **crucial step** in this codebase is the `set_seed(42)` function called at the start of `train_model()`. This locks PyTorch, NumPy, and Python's random number generators. Because the randomness is completely deterministic, every run with identical configuration will yield identical results. This allows ML engineers to run a configuration just once and trust the result.

## Evaluation
The `evaluate.py` script computes **Accuracy**, **Macro F1-Score**, and **Loss** to evaluate how well the model predicts across all 8 classes concurrently (using a `0.5` threshold).

## Logging & Experiment Tracking
Every training run must be logged to **Weights & Biases (W&B)** and also summarized locally.

### 1. Weights & Biases (W&B) Tracking
During training in Colab, the following metrics and configurations are automatically tracked live:
- **Hyperparameters (Config)**: learning rate, epochs, device, model architecture.
- **Epoch-level Metrics**: train loss, val loss, val accuracy, val F1 macro score.
- **Model gradients/parameters**: tracked using `wandb.watch()`.

### 2. Local Experiment Summary
After completion, a summary must be saved to `experiments/latest.json` containing:
1. **Experiment ID**: A unique identifier (date-time hash).
2. **Status**: Training completion status.
3. **Decision**: Keep / Reject.
4. **Metrics**: Best loss, accuracy, and F1 score.
5. **Trained on**: Environment name (e.g., `colab`).
