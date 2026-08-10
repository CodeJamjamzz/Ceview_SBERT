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
