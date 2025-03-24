import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
os.environ['QT_QPA_PLATFORM'] = 'xcb'

'''

Fine Tuning Pipeline for ModernBERT on 6 Class Classification

'''

class EmotionClassifier:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.device = None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
    def check_cuda(self):
        if torch.cuda.is_available():
            self.logger.info(f"CUDA is available: {torch.cuda.is_available()}")
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            self.logger.info(f"CUDA current device: {torch.cuda.current_device()}")
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            self.device = torch.device("cuda")
            return True
        else:
            self.logger.info(f"CUDA is not available: {torch.cuda.is_available()}")
            return False
    
    def initialize_model(self, model_name):
        self.logger.info(f"Model used: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.label_map))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
    
    def load_and_preprocess_data(self, data_path):
        df = pd.read_csv(data_path)

        if 'label' not in df.columns:
            self.logger.error("Error: 'label' column is missing from the dataset.")
            return False

        if df['label'].dtype == 'object':
            df['label'] = df['label'].map(self.reverse_label_map)

        if df['label'].isnull().any():
            self.logger.error("Error: Some labels could not be mapped properly.")
            return False
            
        return df
    
    def plot_dataset_distribution(self, df):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='label', order=range(len(self.label_map)), palette='Set2')
        plt.title('Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Frequency')
        plt.xticks(ticks=range(len(self.label_map)), 
                  labels=[self.label_map[i] for i in range(len(self.label_map))], 
                  rotation=45)
        plt.tight_layout()
        plt.show()
    
    def prepare_datasets(self, df, test_size=0.2, random_state=42, max_length=128):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=test_size,
            random_state=random_state
        )

        self.train_dataset = self.tokenize_and_encode(train_texts, train_labels, max_length)
        self.val_dataset = self.tokenize_and_encode(val_texts, val_labels, max_length)
        
    def tokenize_and_encode(self, texts, labels, max_length):
        encodings = self.tokenizer.batch_encode_plus(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }

        if 'token_type_ids' in encodings:
            dataset_dict['token_type_ids'] = encodings['token_type_ids']
        
        return Dataset.from_dict(dataset_dict)
    
    def compute_metrics(self, p):
        logits, labels = p
        preds = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average=None)  
        precision = precision_score(labels, preds, average='macro', zero_division=1)
        recall = recall_score(labels, preds, average='macro', zero_division=1)

        self.logger.info(f"Accuracy: {accuracy}")
        self.logger.info(f"Precision: {precision}")
        self.logger.info(f"Recall: {recall}")

        emotion_f1_scores = {f"f1_{self.label_map[i]}": f1[i] for i in range(len(f1))}
        for emotion, score in emotion_f1_scores.items():
            self.logger.info(f"{emotion}: {score}")

        metrics = {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
        }

        metrics.update(emotion_f1_scores)
        
        return metrics
    
    def train(self, output_dir, logs_dir, batch_size=32, epochs=3, learning_rate=5e-6):
        if not self.train_dataset or not self.val_dataset:
            self.logger.error("Datasets not prepared. Call prepare_datasets first.")
            return

        loss_callback = LossCallback()
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.01
        )

        outputs_dir = os.path.join(os.getcwd(), output_dir)
        logs_dir = os.path.join(os.getcwd(), logs_dir)

        training_args = TrainingArguments(
            output_dir=outputs_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            learning_rate=learning_rate,
            warmup_steps=500,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=100,
            save_steps=100,
            save_strategy="steps",
            logging_dir=logs_dir,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=True,
            disable_tqdm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[loss_callback, early_stopping]
        )

        trainer.train()
        
        return trainer
    
    def save_model(self, save_dir):
        if not self.model or not self.tokenizer:
            self.logger.error("Model not initialized. Cannot save.")
            return
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self.logger.info(f"Model saved to {save_dir}")


class LossCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])


def main():
    classifier = EmotionClassifier()
    if not classifier.check_cuda():
        print("CUDA not available. Exiting.")
        exit(1)
    model_name = "answerdotai/ModernBERT-base"
    print(f"Model used: {model_name}")
    choice = input("[1] Confirm Training: ")
    
    if choice == "1":
        classifier.initialize_model(model_name)
        df = classifier.load_and_preprocess_data('Trainer & Dataset/Emotion/cleaned.csv')
        if df is False:
            exit(1)
        classifier.plot_dataset_distribution(df)
        classifier.prepare_datasets(df)
        trainer = classifier.train(output_dir='Trainer & Dataset/outputs', logs_dir='Trainer & Dataset/logs')
        classifier.save_model("Trainer & Dataset/models/MBERT6")
        print("Further fine-tuning completed and model saved.")
    else:
        print("Invalid choice. Exiting.")
        exit(1)


if __name__ == "__main__":
    main()
