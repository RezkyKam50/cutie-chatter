import faiss
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import numpy as np
import os

'''

For comparing the cosine similarity between two sentences or paragraph, retrived from encoder model embedding

'''

class TextSimilaritySearch:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(768, 32)
    
    @torch.no_grad()
    def get_embedding(self, texts, model, tokenizer, device, max_len=512):
        # Handle both single text and list of texts
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = tokenizer(texts, 
                         return_tensors="pt", 
                         truncation=True, 
                         padding=True, 
                         max_length=max_len)
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Get the model outputs
        outputs = model(**inputs)
        
        # Use the last hidden state instead of all hidden states
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pooling - take average of all tokens
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sentence_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Convert to numpy and handle batch dimension properly
        embeddings = sentence_embeddings.cpu().numpy()
        return embeddings
    
    def normalize_embeddings(self, embeddings):
        # Normalize along the correct axis
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)
    
    def cosine_similarity(self, embedding_1, embedding_2):
        # Ensure embeddings are normalized
        embedding_1 = self.normalize_embeddings(embedding_1)
        embedding_2 = self.normalize_embeddings(embedding_2)
        
        # Compute dot product
        similarity = np.dot(embedding_1, embedding_2.T)
        return similarity[0][0] if similarity.shape == (1, 1) else similarity
    
    def add_text(self, texts, model, tokenizer, device):
        embedding = self.get_embedding(texts, model, tokenizer, device)
        embedding = self.normalize_embeddings(embedding.astype(np.float32))
        
        # Add to index and metadata
        self.index.add(embedding)
        self.metadata.append({"text": texts})
        return True



if __name__ == "__main__":
    # Initialize the model and tokenizer
    local_directory = os.getcwd()
    model_path = f"{local_directory}/CoreDynamics/models/stardust_6"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,   
        model_max_length=512
    )
    
    model = AutoModel.from_pretrained(
        model_path,
        num_labels=6,
    ).to(device)
    
    # Instantiate the similarity search class
    similarity_search = TextSimilaritySearch(dimension=768)
    
    # Capture the semantic meaning and emotional values by feature-wise values on both of these words
    text1 = "Good job my newphew! i felt really happy today, how about you?"
    text2 = "Nah, he mocked too often"
    
    # Get embeddings
    embedding_1 = similarity_search.get_embedding(text1, model, tokenizer, device)
    embedding_2 = similarity_search.get_embedding(text2, model, tokenizer, device)
    
    # Compute cosine similarity
    similarity = similarity_search.cosine_similarity(embedding_1, embedding_2)
    print(f"Cosine Similarity: {similarity:.4f}")

 
