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
import multiprocessing, ollama, re, sys, os, subprocess


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
        self.current_ai_message.setTextFormat(Qt.TextFormat.RichText)  # Enable rich text
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

    def replace_italic_text(self, text):
        return re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

    def replace_think_tags(self, text):
        text = re.sub(r'<\s*/\s*div\s*>', '</div>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*response\s*>', '</response>', text, flags=re.IGNORECASE)
        text = re.sub(r'---', '<hr>', text)
        text = re.sub(r'<\s*think\s*>', '<span style="font-style: italic; font-weight: 200;">', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '</span><br><br>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/?\s*th?i?n?k?\s*/?>', '', text, flags=re.IGNORECASE)
        return text
    
    def response_only(self, text):
        text = re.sub(r'.*?</think>', '', text, flags=re.DOTALL)
        
        # Existing text cleanup
        text = re.sub(r'<\s*/\s*div\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*response\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'---', '', text)
        text = re.sub(r'<\s*think\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/?\s*th?i?n?k?\s*/?>', '', text, flags=re.IGNORECASE)
        
        return text

    def process_content(self, text):
        if not text:
            return text
        processed = text.replace('\n', '<br>')
        processed = self.replace_italic_text(processed)
        processed = self.replace_think_tags(processed)
        return processed

    def run(self):
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                stream=True,
                options = {
                    "num_thread": self.num_threads,
                    "temperature": 2.56,
                    "top_n": 121,
                    "f16_kv": True,
                    "num_ctx": 1024,
                    "num_batch": 32,
                    "num_prediction": 128
                }
            )
            full_content = ''
            for chunk in stream:
                content = chunk['message']['content']
                full_content += content
                # Process content in worker thread before emitting
                processed_content = self.process_content(content)
                self.chunk_received.emit(processed_content)
                
            # this should be used for sentiment classification
            naked = self.response_only(full_content)

            print(naked)
            self.finished.emit(full_content)

        except Exception as e:
            print(f"Error: {e}")
            error_message = "There seems to be a miscalculation."
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

    def replace_italic_text(self, text):
        return re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

    def replace_think_tags(self, text):
        text = re.sub(r'<\s*/\s*div\s*>', '</div>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*response\s*>', '</response>', text, flags=re.IGNORECASE)
        text = re.sub(r'---', '<hr>', text)
        text = re.sub(r'<\s*think\s*>', '<span style="font-style: italic; font-weight: 200;">', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '</span><br><br>', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/?\s*th?i?n?k?\s*/?>', '', text, flags=re.IGNORECASE)
        return text
    
    def naked_text(self, text):
        text = re.sub(r'<\s*/\s*div\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*response\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'---', '', text)
        text = re.sub(r'<\s*think\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/?\s*th?i?n?k?\s*/?>', '', text, flags=re.IGNORECASE)
        return text
    
    def process_content(self, text):
        if not text:
            return text
        processed = text.replace('\n', '<br>')
        processed = self.replace_italic_text(processed)
        processed = self.replace_think_tags(processed)
        return processed
    


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
                processed_content = self.process_content(content)
                self.chunk_received.emit(processed_content)
            
            self.finished.emit(full_content)

        except Exception as e:
            print(f"Error in OllamaOCRWorker: {e}")
            self.chunk_received.emit(f"Error processing document: {str(e)}")
            self.finished.emit("")



