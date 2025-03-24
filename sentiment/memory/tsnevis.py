import faiss
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

'''

For comparing the cosine similarity between two sentences or paragraph, retrived from encoder model embedding [TSNE PLOT]

'''


class TextSimilaritySearch:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(768, 32)
    
    @torch.no_grad()
    def get_embedding(self, texts, model, tokenizer, device, max_len=512):
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sentence_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sentence_embeddings.cpu().numpy()
    
    def normalize_embeddings(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def cosine_similarity(self, embedding_1, embedding_2):
        embedding_1 = self.normalize_embeddings(embedding_1)
        embedding_2 = self.normalize_embeddings(embedding_2)
        similarity = np.dot(embedding_1, embedding_2.T)
        return similarity[0][0] if similarity.shape == (1, 1) else similarity


if __name__ == "__main__":
    # Load Model and Tokenizer
    local_directory = os.getcwd()
    model_name = f"{local_directory}/CoreDynamics/models/stardust_6"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    similarity_search = TextSimilaritySearch(dimension=768)
    
    # Sample Sentences
    sentences = [
        "I love this movie, it's amazing!",
        "The food was absolutely terrible.",
        "What a fantastic day!",
        "I feel sad and lonely.",
        "Great job on your work!",
        "The weather is awful today.",
        "I am so excited for this event!",
        "This place makes me feel uncomfortable.",
        "I love this food, it's wonderful!"
    ]
    
    # Get Embeddings
    embeddings = np.array([similarity_search.get_embedding(sentence, model, tokenizer, device)[0] for sentence in sentences])
    
    # Reduce Dimensions using t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plot the t-SNE results
    plt.figure(figsize=(10, 6))
    for i, sentence in enumerate(sentences):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.text(reduced_embeddings[i, 0] + 0.1, reduced_embeddings[i, 1] + 0.1, sentence, fontsize=9)
    
    plt.title("t-SNE Visualization of Sentence Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
