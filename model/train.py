"""
Training script logic.
Note: Execution of this should be primarily managed in Colab.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import datetime
import wandb
from .evaluate import evaluate_model
from .model import get_model

def setup_wandb(learning_rate, num_epochs, device):
    """
    Sets up Weights & Biases login and initialization.
    Loads API key from .env if present.
    """
    # Look for .env in current working directory and in the project root relative to this file
    env_paths = [".env", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")]
    for path in env_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        if key.strip() == "WANDB_API_KEY":
                            api_key = val.strip().strip('"').strip("'")
                            if api_key and api_key != "your_api_key_here":
                                os.environ["WANDB_API_KEY"] = api_key
                                break

    try:
        wandb.login()
    except Exception as e:
        print(f"Warning: W&B login failed. If you are offline or in Colab, you may need to authenticate. Error: {e}")

    run = wandb.init(
        entity="jamiel062705-cit-university",
        project="CEVIEW_SBERT",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "device": device,
            "architecture": "SBERT (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) -> Linear(1152, 512) -> ReLU -> Dropout(0.1) -> Linear(512, 8)"
        }
    )
    return run

def train_model(train_dataloader, val_dataloader, num_epochs=20, learning_rate=0.001, device="cpu"):
    print(f"Starting training on {device} for {num_epochs} epochs...")
    model = get_model().to(device)
    
    # Initialize Weights & Biases run
    wandb_run = setup_wandb(learning_rate, num_epochs, device)
    
    # Track the model architecture and gradients
    if wandb_run:
        wandb.watch(model, log="all")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_acc = 0.0
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            desc_ids = batch['desc_input_ids'].to(device)
            desc_mask = batch['desc_attention_mask'].to(device)
            uvp_ids = batch['uvp_input_ids'].to(device)
            uvp_mask = batch['uvp_attention_mask'].to(device)
            serv_ids = batch['services_input_ids'].to(device)
            serv_mask = batch['services_attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(
                desc_input_ids=desc_ids, desc_attention_mask=desc_mask,
                uvp_input_ids=uvp_ids, uvp_attention_mask=uvp_mask,
                services_input_ids=serv_ids, services_attention_mask=serv_mask
            )
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_dataloader)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_dataloader, criterion, device=device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Log epoch-level metrics to W&B
        if wandb_run:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1
            })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            best_f1 = val_f1
            
            # Save the best model weights so you can download/upload them later
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            print("New best model saved to saved_models/best_model.pth!")
    
    # Log results to experiments/latest.json
    experiment_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_data = {
        "experiment_id": experiment_id,
        "status": "completed",
        "decision": "pending",
        "model_type": "sbert",
        "metrics": {
            "train_loss": train_loss,
            "val_loss": best_val_loss,
            "accuracy": best_acc,
            "f1_score": best_f1
        },
        "comparison_to_previous_best": "",
        "trained_on": "colab",
        "next_action": "review"
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/latest.json', 'w') as f:
        json.dump(log_data, f, indent=2)
        
    print(f"Training complete. Results logged to experiments/latest.json")
    
    # Finish the W&B run
    if wandb_run:
        # Also log final summary metrics
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["best_accuracy"] = best_acc
        wandb_run.summary["best_f1_macro"] = best_f1
        wandb_run.finish()
        
    return model
