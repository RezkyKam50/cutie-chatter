from PyQt6.QtWidgets import (
    QVBoxLayout, 
    QWidget, 
    QScrollArea, 
    QFrame, 
    QLabel,
    QGraphicsOpacityEffect,
    QApplication
)
from PyQt6.QtCore import (
    Qt, 
    QPropertyAnimation,
    QObject,
    QMetaObject,
    pyqtSignal
)
from PyQt6.QtCore import (
    QObject,
    pyqtSignal
)
from ocr.docreader import (
    TextExtractor
)
from transformers import TextIteratorStreamer
import multiprocessing, ollama, re, sys, os, subprocess, random, torch
import torch.nn.functional as F
from threading import Thread
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.functional import hessian

'''

    Chat Widget 

'''


class ChatWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.current_ai_message = None
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("chat_frame")
        self.chat_frame = QFrame(self)
        self.chat_frame.setObjectName("chat_frame")
        self.chat_layout = QVBoxLayout(self.chat_frame)
        self.chat_frame.setLayout(self.chat_layout)
        self.scroll_area.setWidget(self.chat_frame)
        self.layout().addWidget(self.scroll_area)

    def add_user_message(self, message):
        
        label = QLabel(f"{message}", self)
        label.setWordWrap(True)
        label.setObjectName("user_response")
        self.chat_layout.addWidget(label)
        self.scroll_to_bottom()

    def start_ai_message(self):
        self.current_ai_message = QLabel(self)
        self.current_ai_message.setWordWrap(True)
        self.current_ai_message.setObjectName("ai_response")
        self.current_ai_message.setTextFormat(Qt.TextFormat.RichText)  
        self.chat_layout.addWidget(self.current_ai_message)
        self.fade_in(self.current_ai_message)
        self.scroll_to_bottom()

    def append_to_ai_message(self, processed_text):
        if not processed_text or not self.current_ai_message:
            return

        current_text = self.current_ai_message.text()
        self.current_ai_message.setText(current_text + processed_text)
        self.scroll_to_bottom()
    
    def scroll_to_bottom(self):
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        self.scroll_area.setObjectName("scroll_area")

    def fade_in(self, widget):
        self.effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(self.effect)
        self.animation = QPropertyAnimation(self.effect, b"opacity")
        self.animation.setDuration(2000)   
        self.animation.setStartValue(0.0)   
        self.animation.setEndValue(1.0)   
        self.animation.start()

'''


    Text Cleaner Class


'''


class TextCleaner():
    def __init__(self, text):
        self.text_to_be_cleaned = text

    def replace_italic_text(self, text):
        return re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

    def replace_think_tags(self, text):
        text = re.sub(r'<\s*/\s*div\s*>', '</think>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*response\s*>', '</think>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*button\s*>', '</think>', text, flags=re.IGNORECASE)
        text = re.sub(r'---', '</think', text)
        text = re.sub(r'<\s*/\s*br\s*>', '</think>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*Compose\s*>', '</think>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*think\s*>', '<span style="font-style: italic; font-weight: 200;">', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '</span><br><br>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/?\s*th?i?n?k?\s*/?>', '', text, flags=re.IGNORECASE)
        return text
    
    def response_only(self, text):
        text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<\s*/\s*div\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*br\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*response\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'---', '', text)
        text = re.sub(r'<\s*think\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/?\s*th?i?n?k?\s*/?>', '', text, flags=re.IGNORECASE)
        return text

    def process_content(self, text):
        if not text:
            return text
        processed = self.text_to_be_cleaned.replace('\n', '<br>')
        processed = self.replace_italic_text(processed)
        processed = self.replace_think_tags(processed)
        return processed
    
'''


    Ollama Worker For direct conversation.


'''

class OllamaWorker(QObject):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, user_message, conversation_history, model_name):
        super().__init__()
        self.user_message = user_message
        self.conversation_history = conversation_history.copy()
        self.num_threads = multiprocessing.cpu_count() / 2
        self.model_name = model_name

    def run(self):
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                stream=True,
                options = {
                    "num_thread": self.num_threads,
                    "temperature": 1.2,
                    "top_n": 50,
                    "top_k": 1.4,
                    "f16_kv": True,
                    "num_ctx": 1024,
                    "num_batch": 32,
                    "num_prediction": 12
                }
            )
            full_content = ''
            for chunk in stream:
                content = chunk['message']['content']
                full_content += content
                string_processor = TextCleaner(content)
                processed_content = string_processor.process_content(string_processor)
                self.chunk_received.emit(processed_content)
                
            naked = string_processor.response_only(full_content)

            print(naked)
            self.finished.emit(full_content)

        except Exception as e:
            print(f"Error: {e}")
            error_message = f"There seems to be a miscalculation.{e}"
            self.chunk_received.emit(error_message)
            self.finished.emit(error_message)


'''


    OCR Worker for Document Extraction : Cons : Multilingual BIAS


'''

class OllamaOCRWorker(QObject):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(str)
    request_file_dialog = pyqtSignal()

    def __init__(self, user_message, conversation_history, model_name):
        super().__init__()
        self.user_message = user_message
        self.conversation_history = conversation_history.copy()
        self.num_threads = multiprocessing.cpu_count() 
        self.text_extractor = TextExtractor()
        self.file_path = None
        self.model_name = model_name

    def set_file_path(self, path):
        self.file_path = path
        if self.file_path:
            self.process_file()

    def run(self):
        self.request_file_dialog.emit()

    def process_file(self):
        try:
            if not self.file_path:
                self.chunk_received.emit("No file selected.")
                self.finished.emit("")
                return
            
            pdf_name = os.path.basename(self.file_path)
            
            if not os.path.exists(self.file_path):
                self.chunk_received.emit("The file does not exist.")
                self.finished.emit("")
                return
            
            extracted_text = self.text_extractor.extract_text_from_pdf(self.file_path)
            
            if not extracted_text:
                self.chunk_received.emit("Failed to extract text from PDF.")
                self.finished.emit("")
                return

            self.conversation_history.append({
                'role': 'user', 
                'content': f"File : {pdf_name}, Content : {extracted_text}"
            })

            stream = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                stream=True,
                options={
                    "num_thread": self.num_threads,
                    "temperature": 2.52,
                    "top_n": 121,
                    "f16_kv": True,
                    "num_ctx": 1024*2,
                    "num_batch": 8,
                    "num_prediction": 1024*2
                }
            )            
            full_content = ''
            for chunk in stream:
                content = chunk['message']['content']
                full_content += content
                string_processor = TextCleaner(content)
                processed_content = string_processor.process_content(content)
                self.chunk_received.emit(processed_content)
            
            self.finished.emit(full_content)

        except Exception as e:
            print(f"Error in OllamaOCRWorker: {e}")
            self.chunk_received.emit(f"Error processing document: {str(e)}")
            self.finished.emit("")



'''


    QWEN WORKER [Research]


'''
class QwenWorker(QObject):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, model, tokenizer, user_message, conversation_history, vector_amplification=48, max_new_tokens=64, context_length=1024, emotion_vectors=None):
        super().__init__()

        self.max_new_tokens = max_new_tokens
        self.context_length = context_length
        self.conversation_history = conversation_history.copy()
        self.user_message = user_message
        self.tokenizer = tokenizer
        self.model = model
        self.model.config.output_attentions = True
        self.device = self.model.device
        self.injection_vectors = emotion_vectors
        self.response_started = False
        self.hook_handles = []
        self.num_amplification = vector_amplification
        self.bias_embedding = None
        if self.injection_vectors is not None:
            self._precompute_bias_embedding()

    def _precompute_bias_embedding(self):
        try:
            amplified_injection_vectors = [vec for vec in self.injection_vectors for _ in range(self.num_amplification)]

            batch_inputs = self.tokenizer(
                amplified_injection_vectors,   
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            input_ids = batch_inputs.input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.base_model(input_ids=input_ids)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                self.bias_embedding = torch.max(embeddings, dim=0)[0]
                print(self.bias_embedding)
                self.bias_embedding = F.normalize(self.bias_embedding, p=2, dim=-1)

        except Exception as e:
            print(f"Error precomputing bias_embedding: {e}")
            self.bias_embedding = None

    def run(self, bias=3.5, hs_scaling=3.5):
        try:
            return self.generate_response(bias, hs_scaling)
        except Exception as e:
            print(f"Error in run: {e}")
            self.finished.emit("")
            return ""
            
    def generate_response(self, bias, hs_scaling):
        self.inputs = self.tokenizer(
            self._format_conversation(self.conversation_history),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_length,
            return_attention_mask=True
        )

        input_ids = self.inputs.input_ids.to(self.device)
        attention_mask = self.inputs.attention_mask.to(self.device)

        if self.injection_vectors is not None and self.bias_embedding is not None:
            self._setup_emotion_hooks(bias, hs_scaling)

        try:
            streamer = TextIteratorStreamer(self.tokenizer)
            
            generation_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'max_new_tokens': self.max_new_tokens,
                'num_return_sequences': 1,
                'pad_token_id': self.tokenizer.pad_token_id,
                'do_sample': True,
                'temperature': 1.0,
                'no_repeat_ngram_size': 5,
                'repetition_penalty': 1.1,
                'streamer': streamer,
                'top_k': 10,
                'top_p': 0.15,
                'early_stopping': True,
                'min_length': 1,
                'forced_bos_token_id': self.tokenizer.bos_token_id,
                'forced_eos_token_id': self.tokenizer.eos_token_id,
                'output_attentions': True
            }
            assistant_pattern = re.compile(r'<\|assistant\|>\n')
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            full_content = []
            for new_text in streamer:
                content = new_text.get('content', '') if isinstance(new_text, dict) else new_text

                if not self.response_started:
                    match = assistant_pattern.search(content)
                    if match:
                        content = content[match.end():]
                        self.response_started = True
                    else:
                        continue

                if content:   
                    full_content.append(content)
                    cleaned_content = self._clean_response(content)
                    if cleaned_content:
                        self.chunk_received.emit(cleaned_content)

            self._cleanup_hooks()
    
            cleaned_full_content = self._clean_response(''.join(full_content))
            self.finished.emit(cleaned_full_content)
            return cleaned_full_content

        except Exception as e:
            print(f"Error in generate_response: {e}")
            self._cleanup_hooks()
            self.finished.emit("")
            return ""

    def compute_pca(self, dataloader, num_samples=1000, variance_threshold=0.95):
        """Compute PCA parameters using hidden states from the dataloader"""
        hidden_states = []
        hook_handles = []
        layers = self.model.model.layers
        target_modules = []
        for i in range(len(layers)):
            if i % 2 == 0:
                target_modules.append(layers[i].mlp)
            else:
                target_modules.append(layers[i].self_attn)

        def hook(module, input, output):
            hidden_states.append(output[0].cpu())   

        for module in target_modules:
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                self.model(inputs)
                if len(hidden_states) >= num_samples:
                    break

        for handle in hook_handles:
            handle.remove()

        hidden_states = torch.cat(hidden_states, dim=0)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))   
        mu = hidden_states.mean(dim=0)
        centered = hidden_states - mu
        cov = torch.matmul(centered.T, centered) / (centered.size(0) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]
        total_var = eigenvalues.sum()
        var_ratio = eigenvalues.cumsum(0) / total_var
        k = (var_ratio < variance_threshold).sum() + 1
        V = eigenvectors[:, :k]
        self.pca_mu = mu.to(self.device)
        self.pca_V = V.to(self.device)

    def _setup_emotion_hooks(self, bias, hs_scaling):
        """Set up emotion injection hooks with PCA-based steering"""
        try:
            bias_embedding = self.bias_embedding.to(dtype=torch.float16)
            bias_embedding = F.normalize(bias_embedding, p=2, dim=-1)
            if not hasattr(self, 'pca_mu') or not hasattr(self, 'pca_V'):
                raise ValueError("PCA parameters not computed. Call compute_pca() first.")

            @torch.no_grad()
            def pca_steering_hook(module, input, output):
                hidden_states = output[0]  
                batch, seq, dim = hidden_states.shape
                h_flat = hidden_states.view(-1, dim)
                centered = h_flat - self.pca_mu
                h_pca = torch.matmul(centered, self.pca_V)
                bias_centered = bias_embedding - self.pca_mu
                bias_pca = torch.matmul(bias_centered, self.pca_V)
                std = h_flat.std(dim=-1, keepdim=True)
                global_std = h_flat.std()
                alpha = torch.sigmoid((std / global_std) * hs_scaling)
                expanded_bias = bias_pca.unsqueeze(0).unsqueeze(0).expand(batch, seq, -1)
                modified_pca = h_pca.view(batch, seq, -1) + alpha * (bias * expanded_bias.to(hidden_states.device))
                reconstructed_centered = torch.matmul(modified_pca.view(-1, modified_pca.size(-1)), self.pca_V.T)
                reconstructed = reconstructed_centered + self.pca_mu
                modified_hidden = reconstructed.view(batch, seq, dim)

                return (modified_hidden,) + output[1:]

            layers = self.model.model.layers
            for i in range(min(len(layers), 24)):  
                target_layer = layers[i]
                hook_target = target_layer.mlp if i % 2 == 0 else target_layer.self_attn
                self.hook_handles.append(hook_target.register_forward_hook(pca_steering_hook))

        except Exception as e:
            print(f"Error in PCA-based emotion setup: {e}")
            self._cleanup_hooks()

    def _cleanup_hooks(self):
        """Clean up all hook handles"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    @torch.no_grad()
    def get_embeddings(self, text):
        """Get embeddings for a single text input"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        outputs = self.model.base_model(input_ids=input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def _format_conversation(self, messages):
        """Format the conversation history"""
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _clean_response(self, text):
        """Clean response text by removing special tokens and emojis"""
        text = re.sub(r'<\|[^|]+\|>', '', text)
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            u"\U0001F700-\U0001F77F"  # Alchemical Symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed Characters
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        
        return text



'''

class QwenWorker(QObject):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, model, tokenizer, user_message, conversation_history, max_new_tokens=64, context_length=1024, emotion_vectors=None):
        super().__init__()

        self.max_new_tokens = max_new_tokens
        self.context_length = context_length
        self.conversation_history = conversation_history.copy()
        self.user_message = user_message
        self.tokenizer = tokenizer
        self.model = model
        self.model.config.output_attentions = True
        self.device = self.model.device
        self.injection_vectors = emotion_vectors
        self.response_started = False
        self.hook_handles = []

        # Precompute bias_embedding if emotion_vectors are provided
        self.bias_embedding = None
        if self.injection_vectors is not None:
            self._precompute_bias_embedding()

    def _precompute_bias_embedding(self):
        try:
            # Duplicate each emotion vector 4 times for amplification
            amplified_injection_vectors = [vec for vec in self.injection_vectors for _ in range(48)]
            
            # Batch process all emotion vectors at once
            batch_inputs = self.tokenizer(
                amplified_injection_vectors,  # Use duplicated vectors
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            input_ids = batch_inputs.input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.base_model(input_ids=input_ids)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                self.bias_embedding = torch.max(embeddings, dim=0)[0]
                print(self.bias_embedding)
                self.bias_embedding = F.normalize(self.bias_embedding, p=2, dim=-1)

        except Exception as e:
            print(f"Error precomputing bias_embedding: {e}")
            self.bias_embedding = None

    def run(self, bias=3.5, hs_scaling=3.5):
        try:
            return self.generate_response(bias, hs_scaling)
        except Exception as e:
            print(f"Error in run: {e}")
            self.finished.emit("")
            return ""
            
    def generate_response(self, bias, hs_scaling):
        self.inputs = self.tokenizer(
            self._format_conversation(self.conversation_history),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_length,
            return_attention_mask=True
        )

        input_ids = self.inputs.input_ids.to(self.device)
        attention_mask = self.inputs.attention_mask.to(self.device)

        if self.injection_vectors is not None and self.bias_embedding is not None:
            self._setup_emotion_hooks(bias, hs_scaling)

        try:
            streamer = TextIteratorStreamer(self.tokenizer)
            
            generation_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'max_new_tokens': self.max_new_tokens,
                'num_return_sequences': 1,
                'pad_token_id': self.tokenizer.pad_token_id,
                'do_sample': True,
                'temperature': 1.0,
                'no_repeat_ngram_size': 5,
                'repetition_penalty': 1.1,
                'streamer': streamer,
                'top_k': 10,
                'top_p': 0.15,
                'early_stopping': True,
                'min_length': 1,
                'forced_bos_token_id': self.tokenizer.bos_token_id,
                'forced_eos_token_id': self.tokenizer.eos_token_id,
                'output_attentions': True
            }

            # Use a cached regular expression for better performance
            assistant_pattern = re.compile(r'<\|assistant\|>\n')
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            full_content = []
            for new_text in streamer:
                content = new_text.get('content', '') if isinstance(new_text, dict) else new_text

                if not self.response_started:
                    match = assistant_pattern.search(content)
                    if match:
                        content = content[match.end():]
                        self.response_started = True
                    else:
                        continue

                if content:   
                    full_content.append(content)
                    # Use _clean_response instead of direct regex substitution
                    cleaned_content = self._clean_response(content)
                    if cleaned_content:
                        self.chunk_received.emit(cleaned_content)

            self._cleanup_hooks()
                
            # Use _clean_response for final cleanup
            cleaned_full_content = self._clean_response(''.join(full_content))
            self.finished.emit(cleaned_full_content)
            return cleaned_full_content

        except Exception as e:
            print(f"Error in generate_response: {e}")
            self._cleanup_hooks()
            self.finished.emit("")
            return ""

    def _setup_emotion_hooks(self, bias, hs_scaling):
        """Set up emotion injection hooks with memoized bias embedding"""
        try:
            # Cache the bias embedding at correct precision to avoid repeated conversions
            bias_embedding_4bit = self.bias_embedding.to(dtype=torch.float16)
            bias_embedding_4bit = F.normalize(bias_embedding_4bit, p=2, dim=-1)

  
 
            @torch.no_grad()
            def attention_hook(module, input, output):

                hidden_states = output[0]  # Shape: (batch_size, seq_len, hidden_dim)

                # Modify hidden states
                expanded_bias = bias_embedding_4bit.unsqueeze(0).unsqueeze(0).expand(
                    hidden_states.size(0), hidden_states.size(1), -1
                )
                alpha = torch.sigmoid((hidden_states.std(dim=-1, keepdim=True) / hidden_states.std()) * hs_scaling)

                modified_hidden = hidden_states + alpha * (bias * expanded_bias.to(hidden_states.device))

                return (modified_hidden,) + output[1:]

            
            @torch.no_grad()
            def mlp_hook(module, input, output):
                alpha = bias * torch.sigmoid(output.mean(dim=-1, keepdim=True))
                expanded_bias = bias_embedding_4bit.unsqueeze(0).unsqueeze(0).expand(
                    output.size(0), output.size(1), -1
                )
                modified_output = output + alpha * expanded_bias.to(output.device)
                return modified_output

            # Apply hooks only to every other layer for efficiency
            layers = self.model.model.layers
            for i in range(0, 27):  # Ensure only indices 0 to 23 are used
                if i >= len(layers):  # Prevent out-of-bounds errors
                    print(f"Layer index out of bound, maximum decoder layer as known : {layers} layers")
                    break
                if i % 2 == 0:
                    self.hook_handles.append(layers[i].mlp.register_forward_hook(mlp_hook))
                else:
                    self.hook_handles.append(layers[i].self_attn.register_forward_hook(attention_hook))

        except Exception as e:
            print(f"Error in Latent Injection setup: {e}")
            self._cleanup_hooks()

    def _cleanup_hooks(self):
        """Clean up all hook handles"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    @torch.no_grad()
    def get_embeddings(self, text):
        """Get embeddings for a single text input"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        outputs = self.model.base_model(input_ids=input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def _format_conversation(self, messages):
        """Format the conversation history"""
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _clean_response(self, text):
        """Clean response text by removing special tokens and emojis"""
        # Remove special tokens like <|assistant|>, <|user|>, etc.
        text = re.sub(r'<\|[^|]+\|>', '', text)
        
        # Remove emojis using a regex pattern that matches a wide range of emojis
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            u"\U0001F700-\U0001F77F"  # Alchemical Symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed Characters
            "]+", flags=re.UNICODE
        )
        
        # Remove emojis from the text
        text = emoji_pattern.sub(r'', text)
        
        return text


'''



