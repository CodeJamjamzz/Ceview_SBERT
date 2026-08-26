import torch
import torch.nn as nn
from transformers import AutoModel

class TourismClassifier(nn.Module):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", hidden_size=512, num_classes=8):
        super(TourismClassifier, self).__init__()
        
        # Load the SBERT backbone
        self.sbert = AutoModel.from_pretrained(model_name)
        
        # Freeze SBERT weights since we only use it as an embedding extractor
        for param in self.sbert.parameters():
            param.requires_grad = False
            
        # MiniLM-L12 has hidden size of 384. Concatenating 3 embeddings -> 384 * 3 = 1152
        sbert_out_dim = self.sbert.config.hidden_size
        concat_dim = sbert_out_dim * 3
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, desc_input_ids, desc_attention_mask, 
                      uvp_input_ids, uvp_attention_mask, 
                      services_input_ids, services_attention_mask):
        
        # We wrap SBERT execution in no_grad to ensure no gradients are computed for the backbone
        # Even though requires_grad=False is set, this saves memory and computation.
        with torch.no_grad():
            # 1. Description embedding
            desc_out = self.sbert(input_ids=desc_input_ids, attention_mask=desc_attention_mask)
            desc_emb = self.mean_pooling(desc_out, desc_attention_mask)
            
            # 2. UVP embedding
            uvp_out = self.sbert(input_ids=uvp_input_ids, attention_mask=uvp_attention_mask)
            uvp_emb = self.mean_pooling(uvp_out, uvp_attention_mask)
            
            # 3. Services embedding
            serv_out = self.sbert(input_ids=services_input_ids, attention_mask=services_attention_mask)
            serv_emb = self.mean_pooling(serv_out, services_attention_mask)
        
        # Concatenate the 3 embeddings
        # Shape: (batch_size, concat_dim)
        concatenated = torch.cat((desc_emb, uvp_emb, serv_emb), dim=1)
        
        # Pass through the classification hidden layers
        logits = self.classifier(concatenated)
        
        # Note: Sigmoid is applied here as this is the second attempt model architecture.
        # The output is now probabilities (0 to 1).
        return logits

def get_model():
    """
    Returns an instance of the TourismClassifier.
    """
    return TourismClassifier()

if __name__ == "__main__":
    # Dummy verification script
    print("Testing TourismClassifier initialization and forward pass...")
    model = get_model()
    
    # Create dummy input tensors for a batch of 2
    dummy_input_ids = torch.randint(0, 1000, (2, 32))
    dummy_attention_mask = torch.ones((2, 32))
    
    logits = model(
        desc_input_ids=dummy_input_ids, desc_attention_mask=dummy_attention_mask,
        uvp_input_ids=dummy_input_ids, uvp_attention_mask=dummy_attention_mask,
        services_input_ids=dummy_input_ids, services_attention_mask=dummy_attention_mask
    )
    
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (2, 8), "Output shape should be (batch_size, 8)"
    print("Test passed successfully!")
