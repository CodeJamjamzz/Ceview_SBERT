"""
Preprocessing logic for business description, value proposition, and services.
Also contains the PyTorch Dataset for loading and tokenizing the JSON data.
"""
import os
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Explicitly map categories to indices
TOURISM_CATEGORIES = [
    "Coastal & Island",              # 0
    "Adventure & Nature",            # 1
    "Cultural & Heritage",           # 2
    "Theme Parks / Entertainment",   # 3
    "Urban & City",                  # 4
    "Culinary & Gastronomy",         # 5
    "Accommodation & Staycation",    # 6
    "OUT_OF_SCOPE"                   # 7
]

CATEGORY_TO_INDEX = {category: idx for idx, category in enumerate(TOURISM_CATEGORIES)}

class TourismDataset(Dataset):
    def __init__(self, data_list, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_length=128):
        """
        data_list: list of dicts with 'description', 'uvp', 'services', 'labels'
        """
        self.data = data_list
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.num_classes = len(TOURISM_CATEGORIES)

    def __len__(self):
        return len(self.data)

    def encode_labels(self, labels):
        # Create a multi-hot encoded tensor
        encoded = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in CATEGORY_TO_INDEX:
                encoded[CATEGORY_TO_INDEX[label]] = 1.0
            else:
                # Fallback to out of scope if unrecognised
                encoded[CATEGORY_TO_INDEX["OUT_OF_SCOPE"]] = 1.0
        
        # If no labels were provided, map to out of scope
        if len(labels) == 0:
            encoded[CATEGORY_TO_INDEX["OUT_OF_SCOPE"]] = 1.0
            
        return encoded

    def __getitem__(self, idx):
        item = self.data[idx]
        
        description = item.get("description", "")
        uvp = item.get("uvp", "")
        services_list = item.get("services", [])
        services = ", ".join(services_list)
        labels = item.get("labels", [])
        
        # Tokenize Description
        desc_enc = self.tokenizer(
            description, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        # Tokenize UVP
        uvp_enc = self.tokenizer(
            uvp, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        # Tokenize Services
        serv_enc = self.tokenizer(
            services, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        # Multi-hot encode labels
        label_tensor = self.encode_labels(labels)
        
        return {
            "desc_input_ids": desc_enc["input_ids"].squeeze(0),
            "desc_attention_mask": desc_enc["attention_mask"].squeeze(0),
            "uvp_input_ids": uvp_enc["input_ids"].squeeze(0),
            "uvp_attention_mask": uvp_enc["attention_mask"].squeeze(0),
            "services_input_ids": serv_enc["input_ids"].squeeze(0),
            "services_attention_mask": serv_enc["attention_mask"].squeeze(0),
            "labels": label_tensor
        }

def load_data_from_json(data_dir: str):
    """
    Reads all JSON files in the given directory and aggregates the elements.
    Expected format per file: A JSON array of objects.
    """
    all_data = []
    search_pattern = os.path.join(data_dir, "*.json")
    for file_path in glob.glob(search_pattern):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    all_data.extend(file_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return all_data

def get_dataloaders(data_dir="data", batch_size=16, shuffle=True):
    """
    Helper function to load data and return a PyTorch DataLoader.
    """
    data_list = load_data_from_json(data_dir)
    dataset = TourismDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    # Test script to verify the logic works
    print("Testing data loading and preprocessing...")
    # Using 'data' assuming it runs from project root
    dataset_list = load_data_from_json("data")
    print(f"Loaded {len(dataset_list)} items.")
    
    if len(dataset_list) > 0:
        dataset = TourismDataset(dataset_list)
        sample = dataset[0]
        print(f"Sample labels tensor: {sample['labels']}")
        print(f"Sample desc input ids shape: {sample['desc_input_ids'].shape}")
        print("Preprocessing test passed!")
