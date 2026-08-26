# Integration Contract

## Division of Responsibility
- **Model**: Converts input text (description, value prop, services) into an array/dictionary of 8 scores (7 categories + 1 OUT_OF_SCOPE). The `predict_business` function returns a dictionary where keys are the category names and values are the boolean flags indicating if they exceed the threshold.

> [!NOTE]
> Under the hood, the model returns a 1x8 tensor. The backend should map these indices strictly: `0`=Coastal & Island, `1`=Adventure & Nature, `2`=Cultural & Heritage, `3`=Theme Parks / Entertainment, `4`=Urban & City, `5`=Culinary & Gastronomy, `6`=Accommodation & Staycation, and `7`=OUT_OF_SCOPE.

- **Backend**: Validates the input before sending it to the model. Applies the threshold to the scores to determine the final selected categories. Handles API errors and rate limits.
- **Training**: Training does *not* happen in the backend. The backend only uses a frozen, exported version of the model for inference.

## Example Request
```json
{
  "business_description": "We are a boutique hotel located in the heart of the city.",
  "unique_value_proposition": "Luxury stay with an authentic local experience.",
  "list_of_services": ["Accommodation", "Breakfast", "Guided City Tours"]
}
```

## Example Response
```json
{
  "scores": {
    "Coastal & Island": 0.05,
    "Adventure & Nature": 0.10,
    "Cultural & Heritage": 0.02,
    "Theme Parks / Entertainment": 0.01,
    "Urban & City": 0.15,
    "Culinary & Gastronomy": 0.30,
    "Accommodation & Staycation": 0.95,
    "OUT_OF_SCOPE": 0.01
  }
}
```
*(The backend applies a threshold, e.g. `0.5`, resulting in `["Accommodation", "Travel Services"]`)*
*(The backend applies a threshold, e.g. `0.5`, resulting in `["Accommodation", "Travel Services"]`)*

## Loading the Model in the Backend
When you download `best_model.pth` from Colab, you are downloading a dictionary of raw numbers (weights). The `.pth` file **does not** contain the structure/code of the model. To use it in a backend (like FastAPI, Flask, or Django), you must follow these steps:

### 1. Recreate the Architecture (sbert-unfrozen-final)
Your backend needs the literal Python code that defines the `TourismClassifier` so it knows how to structure the "brain" before loading the weights. 

> [!NOTE]
> This specific architecture block corresponds exactly to the **"sbert-unfrozen-final"** run in Weights & Biases (which is the "Second Attempt" architecture configuration). You must use the `.pth` file downloaded from that exact training run.

**Copy and paste this exact code into your backend (e.g., `model.py`):**
```python
import torch
import torch.nn as nn
from transformers import AutoModel

class TourismClassifier(nn.Module):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", hidden_size=512, num_classes=8):
        super(TourismClassifier, self).__init__()
        
        # Load the SBERT backbone
        self.sbert = AutoModel.from_pretrained(model_name)
        
        # MiniLM-L12 has hidden size of 384. Concatenating 3 embeddings -> 384 * 3 = 1152
        sbert_out_dim = self.sbert.config.hidden_size
        concat_dim = sbert_out_dim * 3
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, desc_input_ids, desc_attention_mask, uvp_input_ids, uvp_attention_mask, services_input_ids, services_attention_mask):
        with torch.no_grad():
            desc_out = self.sbert(input_ids=desc_input_ids, attention_mask=desc_attention_mask)
            desc_emb = self.mean_pooling(desc_out, desc_attention_mask)
            
            uvp_out = self.sbert(input_ids=uvp_input_ids, attention_mask=uvp_attention_mask)
            uvp_emb = self.mean_pooling(uvp_out, uvp_attention_mask)
            
            serv_out = self.sbert(input_ids=services_input_ids, attention_mask=services_attention_mask)
            serv_emb = self.mean_pooling(serv_out, services_attention_mask)
        
        concatenated = torch.cat((desc_emb, uvp_emb, serv_emb), dim=1)
        return self.classifier(concatenated)

def get_model():
    return TourismClassifier()
```

Then, in your main backend script, instantiate it:
```python
# Import the function you just copy-pasted
from model import get_model 
import torch

device = "cpu" # Backend usually runs on CPU for inference
model = get_model().to(device)
```

### 2. Inject the Weights
Load the `.pth` file and inject its numbers into the empty model, then put the model in evaluation mode (this disables layers like Dropout that are only used during training):
```python
model.load_state_dict(torch.load("path/to/your/best_model.pth", map_location=device))
model.eval()
```
> [!IMPORTANT]  
> This loading step should only happen **once** when your backend server starts up! Do not load the model inside your API route, or every single request will take several seconds to process.

### 3. Tokenize Incoming Requests
When a JSON request hits your API, you must tokenize the raw text strings exactly how they were tokenized during training using the HuggingFace tokenizer:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Tokenize Description
desc_tokens = tokenizer(
    request.business_description,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
# Repeat for UVP and Services...
```

### 4. Run Inference
Pass the tokenized inputs into the model to get your predictions:
```python
with torch.no_grad(): # Tells PyTorch not to calculate gradients (saves memory/time)
    logits = model(
        desc_input_ids=desc_tokens['input_ids'], 
        desc_attention_mask=desc_tokens['attention_mask'],
        uvp_input_ids=uvp_tokens['input_ids'], 
        uvp_attention_mask=uvp_tokens['attention_mask'],
        services_input_ids=serv_tokens['input_ids'], 
        services_attention_mask=serv_tokens['attention_mask']
    )
    
    # Threshold the output probabilities
    predictions = (logits > 0.5).float()
```
