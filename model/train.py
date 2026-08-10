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
from .evaluate import evaluate_model
from .model import get_model

def train_model(train_dataloader, val_dataloader, num_epochs=20, learning_rate=0.001    , device="cpu"):
    print(f"Starting training on {device} for {num_epochs} epochs...")
    model = get_model().to(device)
    
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
    return model
