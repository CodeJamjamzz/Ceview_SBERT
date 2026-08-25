"""
Evaluation logic.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, dataloader, criterion, device="cpu"):
    """
    Evaluates the model on the provided dataloader.
    Returns validation loss, accuracy, and macro F1 score.
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            desc_ids = batch['desc_input_ids'].to(device)
            desc_mask = batch['desc_attention_mask'].to(device)
            uvp_ids = batch['uvp_input_ids'].to(device)
            uvp_mask = batch['uvp_attention_mask'].to(device)
            serv_ids = batch['services_input_ids'].to(device)
            serv_mask = batch['services_attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            logits = model(
                desc_input_ids=desc_ids, desc_attention_mask=desc_mask,
                uvp_input_ids=uvp_ids, uvp_attention_mask=uvp_mask,
                services_input_ids=serv_ids, services_attention_mask=serv_mask
            )
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            # Since the model applies Sigmoid internally now, just threshold the output
            preds = (logits > 0.5).float()
            
            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
            
    avg_loss = total_loss / len(dataloader)
    
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return avg_loss, acc, f1
