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
3. Install dependencies: `pip install -r requirements.txt` (or manually install `torch`, `transformers`, `scikit-learn`).

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

## Evaluation
The `evaluate.py` script computes **Accuracy**, **Macro F1-Score**, and **Loss** to evaluate how well the model predicts across all 8 classes concurrently (using a `0.5` threshold).

## Logging Requirements
Every training run in Colab must log the following fields to ensure reproducibility and track progress:
1. **Experiment ID**: A unique identifier (e.g., date-time hash).
2. **Git Commit**: The commit hash of the code used for training.
3. **Config**: Hyperparameters used (learning rate, batch size, epochs, threshold).
4. **Dataset Info**: Version or size of the training/validation data.
5. **Metrics**: Train/val/test loss, accuracy, F1 score (macro/micro for multi-label).
6. **Decision**: Keep / Reject / Deploy.
7. **Notes**: Any observations from the run.

These results should be summarized and saved to `experiments/latest.json`.
