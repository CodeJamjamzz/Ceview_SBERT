# BLOCKERS.md

This file lists tasks or domains that are out of scope for the AI agent to solve alone.

- **Infrastructure Limits**: Do not attempt to run full model training locally in this repository. Training must be done on Google Colab.
- **External Accounts**: The agent cannot authenticate to Google Colab, Hugging Face, or Weights & Biases on behalf of the user. The user must handle authentication.
- **Production Deployment**: Do not attempt to automatically deploy this model to a production server unless explicitly paired with the user.
- **GitHub File Limits**: Model weights (`*.pth`, `saved_models/`) must **NEVER** be committed or pushed to GitHub due to standard 100MB file limits. They should be hosted elsewhere (e.g., HuggingFace, S3) and downloaded.
