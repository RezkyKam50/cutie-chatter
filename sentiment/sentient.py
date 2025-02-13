import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from typing import Dict, Optional
from dataclasses import dataclass

CompositeDictionary = {
    "bittersweet": {
        'emotions': [0, 1, 2],  # sadness + joy + love
        'weights': [0.615, 0.615, 0.525]  # moderate amplification
    },
    "psychological anguish": {
        'emotions': [0, 2, 3],  # sadness + love + anger
        'weights': [0.785, 0.615, 0.35]  # moderate amplification
    },
    "frantic fear response": {
        'emotions': [0, 3, 4],  # sadness + anger + fear
        'weights': [0.525, 0.70, 0.525]  # moderate amplification
    },
    "cognitive disillusionment": {
        'emotions': [0, 1, 5],  # sadness + joy + surprise
        'weights': [0.70, 0.44, 0.615]  # moderate amplification
    },
    "euphoric passion": {
        'emotions': [1, 2, 4],  # joy + love + fear
        'weights': [0.70, 0.70, 0.35]  # moderate amplification
    },
    "ambivalent joy": {
        'emotions': [1, 3, 4],  # joy + anger + fear
        'weights': [0.785, 0.525, 0.4375]  # moderate amplification
    },
    "conflicted happiness": {
        'emotions': [0, 1, 3],  # sadness + joy + anger
        'weights': [0.525, 0.7875, 0.4375]  # moderate amplification
    },
    "bittersweet contentment": {
        'emotions': [1, 0, 2],  # joy + sadness + love
        'weights': [0.70, 0.70, 0.35]  # moderate amplification
    },
    "state of nervous tension": {
        'emotions': [1, 4, 5],  # joy + fear + surprise
        'weights': [0.4375, 0.70, 0.525]  # moderate amplification
    },
    "nostalgic affection": {
        'emotions': [0, 1, 2],  # sadness + joy + love
        'weights': [0.525, 0.525, 0.70]  # moderate amplification
    },
    "wounded indignation": {
        'emotions': [0, 3, 4],  # sadness + anger + fear
        'weights': [0.785, 0.615, 0.35]  # moderate amplification
    },
    "aesthetic amazement": {
        'emotions': [1, 2, 5],  # joy + love + surprise
        'weights': [0.70, 0.4375, 0.615]  # moderate amplification
    },
    "intense romantic passion": {
        'emotions': [2, 3, 4],  # love + anger + fear
        'weights': [0.525, 0.875, 0.35]  # moderate amplification
    },
    "tentative hope": {
        'emotions': [1, 4, 5],  # joy + fear + surprise
        'weights': [0.6125, 0.70, 0.4375]  # moderate amplification
    },
    "careful compassion": {
        'emotions': [1, 2, 4],  # joy + love + fear
        'weights': [0.70, 0.6125, 0.4375]  # moderate amplification
    },
    "shock-induced devastation": {
        'emotions': [0, 4, 5],  # sadness + fear + surprise
        'weights': [0.875, 0.4375, 0.4375]  # moderate amplification
    }
}

@dataclass
class EmotionPrediction:
    dominant_primary_emotion: str
    dominant_primary_logits: float
    primary_emotion_logits: Dict[str, float]
    dominant_composite_emotion: str
    dominant_composite_logits: float
    top_n_composite_emotions: Dict[str, float]

class EmotionClassifier:
    def __init__(self,
                model: any, 
                tokenizer: any, 
                device: any,
                composite_dictionary: Optional[Dict] = None
                ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device 
        
        self.emotion_to_label = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        self.composite_dictionary = composite_dictionary or {}

        self.composite_cache = {
            name: (
                torch.tensor(data['emotions'], device=self.device),
                torch.tensor(data['weights'], device=self.device, dtype=torch.float32)
            )
            for name, data in self.composite_dictionary.items()
        }

    @torch.no_grad()  
    def GetEmotionForClassification(
        self,
        texts: str,
        threshold: float = 0,
        temperature: float = 1.0,
        max_length: int = 512,
        top_n: int = 5
    ) -> EmotionPrediction:
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
                return_token_type_ids=False,  
                return_attention_mask=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logits = self.model(**inputs)[0]
            
            logits = logits / temperature

            logits_np = logits.cpu().numpy()[0]

            emotion_logits = {
                self.emotion_to_label[i]: float(logit)
                for i, logit in enumerate(logits_np)
            }
            
            filtered_emotions = {
                emotion: logit 
                for emotion, logit in emotion_logits.items() 
                if logit >= threshold
            }
            
            dominant_emotion, max_logit = max(
                filtered_emotions.items(),
                key=lambda x: x[1],
                default=("neutral", 0)
            )
            
            composite_logits = {}
            for name, (indices, weights) in self.composite_cache.items():
                selected_logits = logits.index_select(1, indices)
                score = (selected_logits * weights).sum().item()
                composite_logits[name] = score
            
            filtered_composites = {
                k: v for k, v in composite_logits.items() 
                if v >= threshold
            }
            
            sorted_composites = dict(
                sorted(
                    filtered_composites.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
            )
            
            dominant_composite = max(
                sorted_composites.items(),
                key=lambda x: x[1],
                default=("neutral", 0)
            )
            
            return EmotionPrediction(
                dominant_primary_emotion=dominant_emotion,
                dominant_primary_logits=max_logit,
                primary_emotion_logits=emotion_logits,
                dominant_composite_emotion=dominant_composite[0],
                dominant_composite_logits=dominant_composite[1],
                top_n_composite_emotions=sorted_composites
            )    
        except Exception as e:
            print(f"Error in emotion classification: {str(e)}")
            return EmotionPrediction(
                dominant_primary_emotion="Error",
                dominant_primary_logits=0.0,
                primary_emotion_logits={},
                dominant_composite_emotion="Error",
                dominant_composite_logits=0.0,
                top_n_composite_emotions={}
            )
        

def L2S(logits: torch.Tensor) -> Dict[str, float]:
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(logits).to(logits.device).numpy()[0]
    
    emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    
    return {
        emotion_labels[i]: float(prob) 
        for i, prob in enumerate(probabilities)
    }

# Example Use

if __name__ == "__main__":
    
    local_directory = os.getcwd()

    model_path = f"{local_directory}/CoreDynamics/models/stardust_6"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,   
        model_max_length=512
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=6,
        torchscript=True,   
        return_dict=False   
    ).to(device)




    classifier = EmotionClassifier(
        model,
        tokenizer,
        device,
        composite_dictionary=CompositeDictionary
    )
    
    result = classifier.GetEmotionForClassification(
        "I feel so down today, and my mom asked me to wash the dishes",
        threshold=0.2,
        temperature=2.9,  # You can adjust the temperature value here
        max_length=512,
        top_n=3
    )



    print("Primary Emotion Logits:")
    print(result.primary_emotion_logits)
    print("\nTop Composite Emotions:")
    print(result.top_n_composite_emotions)
    
    # Extract values from the dictionary and convert to tensor for probability processing
    logits_values = list(result.primary_emotion_logits.values())
    logits_tensor = torch.tensor(logits_values).unsqueeze(0)  
    probabilities = L2S(logits_tensor)

    print("\nLogits with Softmax:")
    print(probabilities)
