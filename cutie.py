from PyQt6.QtWidgets import (
    QMainWindow,
    QHBoxLayout, 
    QPushButton, 
    QVBoxLayout, 
    QTextEdit,  
    QWidget, 
    QSpacerItem, 
    QSizePolicy
)
from PyQt6.QtCore import (
    Qt, 
    QSize,
    QThread,
    QPropertyAnimation,
    QEasingCurve
)
from PyQt6.QtGui import (
    QIcon
)
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModel,
    AutoTokenizer
)
from backends import (
    ChatWidget, 
    OllamaWorker, 
    OllamaOCRWorker
)
from ocr.docreader import (
    FileSelector
)
from sentiment.sentient import (
    EmotionClassifier,
    L2S
)
from sentiment.memory.textsimilarity import TextSimilaritySearch
import os, screeninfo, torch, re


'''

    Main Process

'''

class CutieTheCutest(QMainWindow):
    def __init__(self, model_name):
        super().__init__()
        self.setWindowTitle("CutieChatter")


        # automatically get available monitors 'in case you have multiple' monitors : where to show
        self.screen_dimension = screeninfo.get_monitors()
        self.last_screen_dimension = self.screen_dimension[-1]
        WIN_W, WIN_H = self.last_screen_dimension.width, self.last_screen_dimension.height
        self.setGeometry(100, 100, int(WIN_W / 2), int(WIN_H / 2.5))
        icon_dir = os.path.dirname(os.path.abspath(__file__))
        self.chat_widget = ChatWidget()

        # title bar setup
        title_bar = QWidget(self)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 15, 0, 0)

        # document button  object
        self.document_button = QPushButton(self)
        self.document_button.setObjectName("document_button")
        icon_document_path = os.path.join(icon_dir, "icons", "document.svg")
        self.document_button.setIcon(QIcon(icon_document_path))
        self.document_button.setIconSize(QSize(34, 44))
        title_layout.addWidget(self.document_button)
        self.document_button.clicked.connect(self.on_submit_ocr)

        # microphone button object
        self.mic_button = QPushButton(self)
        self.mic_button.setObjectName("mic_button")
        icon_document_path = os.path.join(icon_dir, "icons", "mic.svg")
        self.mic_button.setIcon(QIcon(icon_document_path))
        self.mic_button.setIconSize(QSize(34, 44))
        title_layout.addWidget(self.mic_button)

        # spacer exit, minimize button
        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title_layout.addSpacerItem(spacer)

        # minimize button object
        self.minimize_button = QPushButton("minimize", self)
        self.exit_button = QPushButton("exit", self)
        self.minimize_button.setObjectName("minimize_button")
        title_layout.addWidget(self.minimize_button)
        self.exit_button.setObjectName("exit_button")
        title_layout.addWidget(self.exit_button)

        # minimize, exit button backend function
        self.minimize_button.clicked.connect(self.minimize_window)
        self.exit_button.clicked.connect(self.close_window)

        # title bar object
        title_bar.setMinimumHeight(60)
        central_widget = QWidget(self)

        # center widget for proper positioning
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(title_bar, alignment=Qt.AlignmentFlag.AlignTop)

        # chat widget
        main_layout.addWidget(self.chat_widget)

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(10, 10, 10, 10)

        # prompt box object
        self.prompt_box = QTextEdit(self)
        self.prompt_box.setPlaceholderText("Message DeepSeek")
        self.prompt_box.setObjectName("prompt_box")
        self.prompt_box.setMinimumHeight(int(WIN_H * 0.06))
        self.prompt_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chat_widget.setMinimumHeight(int(WIN_H / 1.8))

        # submit button object
        self.submit_button = QPushButton(self)
        self.submit_button.setObjectName("submit_button")
        icon_send_path = os.path.join(icon_dir, "icons", "send.svg")
        self.submit_button.setIcon(QIcon(icon_send_path))
        self.submit_button.setIconSize(QSize(34, 34))

        # submit button backend function
        self.submit_button.clicked.connect(self.on_submit)

        # prompt, submit add to pre defined layout
        input_layout.addWidget(self.prompt_box)
        input_layout.addWidget(self.submit_button)

        # add input layout to main layout
        main_layout.addLayout(input_layout)

        # call {theme}.css       
        self.apply_theme()

        # load sentiment model
        self.model_name = model_name
        self.get_loc_dir = os.getcwd()
        self.model_path = f"{self.get_loc_dir}/CoreDynamics/models/stardust_6"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,   
            model_max_length=512
        )
        self.ModelForSentimentScoring = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=6,
            torchscript=True,   
            return_dict=False   
        ).to(self.device)

        self.classifier = EmotionClassifier(
            self.ModelForSentimentScoring,
            self.tokenizer,
            self.device,
            composite_dictionary=None
        )

        self.ModelForCS = AutoModel.from_pretrained(
            self.model_path,
            num_labels=6,
        ).to(self.device)



        # Vector Memory for Attachment Mechanism Purposes
        self.ai_features_metadata = []
        self.ai_text_metadata = []
        self.user_features_metadata = []
        self.user_text_metadata = []

        self.cosine_of_text_metadata = []
        

        # Initialize these properties
        self.ollama_thread = None
        self.ollama_worker = None
        self.file_selector = FileSelector()

        # Initialize first time command :
        # Role : Assistant, the Assistant should provide guidance to the AI model
        self.conversation_history = [
            {
                'role': 'system',
                'content': """You're CutieChatter, your directive should be :
                            1. Playful.
                            2. Personally engaging with the user.
                            3. Maintain a consistent personal tone.
                            4. Avoid responding using double quotation marks.
                            5. Generate natural responses.
                            """
            }
        ]
        self.is_ollama_thread_running = False

    # minimize button -> call backends of os (window state)
    def minimize_window(self):
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMinimized)

    def close_window(self):
        self.close()


    # backend for submit
    def on_submit(self):
        entered_text = self.prompt_box.toPlainText().strip()
        if entered_text:
            self.animate_submit_button()

            self.chat_widget.add_user_message(entered_text)

            print(entered_text)
            self.user_sentiment_score = self.GetSentimentOnPrimary(entered_text)
            print(f"User Sentiment Score : {self.user_sentiment_score}")   

            # Vector DB
            self.user_text_metadata.append(entered_text)
            self.user_features_metadata.append(self.user_sentiment_score)

            self.prompt_box.clear()
            self.chat_widget.start_ai_message()

            # add the user message to the conversation history
            self.conversation_history.append({'role': 'user', 'content': entered_text})

            self.run_ollama_chat()

    def run_ollama_chat(self):
        if self.is_ollama_thread_running:
            return

        # Disconnect previous worker signals
        if self.ollama_worker:
            try:
                self.ollama_worker.chunk_received.disconnect()
                self.ollama_worker.finished.disconnect()
            except:
                pass

        # Rest of existing code remains the same...
        self.is_ollama_thread_running = True
        self.submit_button.setEnabled(False)

        self.ollama_thread = QThread()
        self.ollama_worker = OllamaWorker(
                                          self.conversation_history[-1]['content'], 
                                          self.conversation_history, 
                                          self.model_name
                                          )
        
        self.ollama_worker.moveToThread(self.ollama_thread)

        # Connect fresh signals
        self.ollama_worker.chunk_received.connect(self.chat_widget.append_to_ai_message)
        self.ollama_worker.finished.connect(self.on_ollama_response_complete)
        self.ollama_thread.started.connect(self.ollama_worker.run)
        self.ollama_thread.start()

    def naked_text(self, text):
        text = re.sub(r'<\s*/\s*div\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*response\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'---', '', text)
        text = re.sub(r'<\s*think\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/?\s*th?i?n?k?\s*/?>', '', text, flags=re.IGNORECASE)
        return text
    
    def on_ollama_response_complete(self, response):
        # add the AI's response to the conversation history
        self.conversation_history.append({'role': 'assistant', 'content': response})

        self.ai_sentiment_score = self.GetSentimentOnPrimary(response)
        print(f"AI Sentiment Score : {self.ai_sentiment_score}")
        

        # Vector DB
        self.ai_text_metadata.append(response)
        self.GetSimilarityPerTurns()
        self.ai_features_metadata.append(self.ai_sentiment_score) 
        
        self.is_ollama_thread_running = False
        self.submit_button.setEnabled(True)
        if self.ollama_worker:
            self.ollama_worker.chunk_received.disconnect()
            self.ollama_worker.finished.disconnect()
        self.ollama_thread.quit()
        self.ollama_worker.deleteLater()


    def on_submit_ocr(self):
        entered_text = self.prompt_box.toPlainText().strip()
        if entered_text:
            self.chat_widget.add_user_message(f"{entered_text}")
            self.prompt_box.clear()
            self.chat_widget.start_ai_message()
            
            self.run_ocr_chat()

    def run_ocr_chat(self):
        if self.is_ollama_thread_running:
            return

        # Disconnect previous signals
        if self.ollama_worker:
            try:
                self.ollama_worker.chunk_received.disconnect()
                self.ollama_worker.finished.disconnect()
                self.ollama_worker.request_file_dialog.disconnect()
            except:
                pass

        # Rest of existing cleanup code...
        self.is_ollama_thread_running = True
        self.document_button.setEnabled(False)

        if self.ollama_thread is not None:
            self.ollama_thread.quit()
            self.ollama_thread.wait()
            self.ollama_thread.deleteLater()
        
        # New worker setup
        self.ollama_thread = QThread()
        self.ollama_worker = OllamaOCRWorker(
                                            self.conversation_history[-1]['content'], 
                                            self.conversation_history,
                                            self.model_name
                                            )
        # Connect fresh signals
        self.ollama_worker.chunk_received.connect(self.chat_widget.append_to_ai_message)
        self.ollama_worker.finished.connect(self.on_ollama_ocr_complete)
        self.ollama_worker.request_file_dialog.connect(self.show_file_dialog)
        self.ollama_thread.started.connect(self.ollama_worker.run)
        
        self.ollama_thread.start()

    def on_ollama_ocr_complete(self, response):
        """Handle OCR completion"""
        if response:  
            self.conversation_history.append({'role': 'assistant', 'content': response})
            self.print_conversation_history()
        self.is_ollama_thread_running = False
        self.document_button.setEnabled(True)

        if self.ollama_worker:
            self.ollama_worker.chunk_received.disconnect()
            self.ollama_worker.finished.disconnect()
        self.ollama_thread.quit()
        self.ollama_worker.deleteLater()

    def show_file_dialog(self):
        """Show file dialog and send path to worker"""
        file_path = self.file_selector.open_file_dialog()
        if self.ollama_worker is not None:
            self.ollama_worker.set_file_path(file_path)

    def print_conversation_history(self):
        """Prints the entire conversation history to console"""
        print("\n=== Conversation History ===")
        for msg in self.conversation_history:
            role = msg['role'].upper()
            content = msg['content'][:70] + "..." if len(msg['content']) > 70 else msg['content']
            print(f"{role}: {content}")
        print("============================\n")

    '''
    
        Attachment Theory
    
    '''
    def GetSentimentOnPrimary(self, text):
        sentiment_result = self.classifier.GetEmotionForClassification(
            texts=self.naked_text(text),
            threshold=0.1,
            temperature=2.0,  
            max_length=512,
            top_n=3
        )
        logits_values = list(sentiment_result.primary_emotion_logits.values())
        logits_tensor = torch.tensor(logits_values).unsqueeze(0)  
        probabilities = L2S(logits_tensor)
        
        return probabilities

    def GetSimilarityPerTurns(self):
        
        recall = TextSimilaritySearch(dimension=768)
        # Turn based logic : User first, AI last. User need to talk first before the AI could response
        if not self.ai_text_metadata:  # If AI text dataset is empty, we can't compare
            return
        embedding_0 = recall.get_embedding(self.user_text_metadata[-1], 
                                           self.ModelForCS, 
                                           self.tokenizer, 
                                           self.device
                                           )
        embedding_1 = recall.get_embedding(self.ai_text_metadata[-1],
                                           self.ModelForCS,
                                           self.tokenizer,
                                           self.device
                                           )
        similarity = recall.cosine_similarity(embedding_0, embedding_1)

        self.cosine_of_text_metadata.append(similarity)
        print(f"Cosine Similarity On Present Dialogue Sequence : {similarity}\n")

        return similarity

    '''
    Apply Theme
    '''
    def apply_theme(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        css_path = os.path.join(script_dir, "themes", "default.css")
        try:
            with open(css_path, "r") as css_file:
                self.setStyleSheet(css_file.read())
        except FileNotFoundError:
            print(f"Theme file not found: {css_path}")
    '''
    Animations
    '''

    def animate_submit_button(self):
        # Create a QPropertyAnimation object
        self.animation = QPropertyAnimation(self.submit_button, b"iconSize")

        # Set the duration of the animation (in milliseconds)
        self.animation.setDuration(1000)

        # Set the easing curve for the animation
        self.animation.setEasingCurve(QEasingCurve.Type.CosineCurve)

        # Set the start and end values for the animation
        self.animation.setStartValue(QSize(34, 34))
        self.animation.setEndValue(QSize(26, 26))

        # Start the animation
        self.animation.start()

        # Connect the finished signal to a slot that resets the icon size
        self.animation.finished.connect(self.reset_submit_button_animation)

    def reset_submit_button_animation(self):
        # Reset the icon size to its original size
        self.submit_button.setIconSize(QSize(34, 34))


 
 


 

 
