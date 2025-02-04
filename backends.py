from PyQt6.QtWidgets import (
    QVBoxLayout, 
    QWidget, 
    QScrollArea, 
    QFrame, 
    QLabel,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    Qt, 
    QPropertyAnimation,
    QObject,
    pyqtSignal
)
from PyQt6.QtCore import (
    QObject,
    pyqtSignal
)
import multiprocessing, ollama, re

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

    def append_to_ai_message(self, text):
        if not text or not self.current_ai_message:
            return  # avoid unnecessary processing if text is empty or no current message

        # eeplace newline characters with <br> tags
        processed_text = text.replace('\n', '<br>')

        # process italic text (handle paired asterisks)
        processed_text = self.replace_italic_text(processed_text)

        # process <think> tags with styled spans
        processed_text = self.replace_think_tags(processed_text)

        # append the processed text to the current AI message
        current_text = self.current_ai_message.text()
        self.current_ai_message.setText(current_text + processed_text)

        self.scroll_to_bottom()
        
    # the function below only applies to deepseek reasoning generation, other models will dynamically adjust by ignoring this function
    def replace_italic_text(self, text):
        # Use a regex to replace anything between * and * with <em></em>
        # This ensures all paired * symbols are correctly handled
        return re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)


    def replace_think_tags(self, text):
        """
        Replace <think> tags with styled span tags, properly handle </div> and </response> tags, 
        and add a condition to handle '---' for horizontal rules.
        Handles potential malformed or partially rendered tags.
        
        Args:
            text (str): Input text containing <think>, </div>, </response>, and/or '---'
            
        Returns:
            str: Text with replaced tags
        """
        # make sure </div> is preserved exactly as is
        text = re.sub(r'<\s*/\s*div\s*>', '</div>', text, flags=re.IGNORECASE)

        # make sure also  </response> is preserved exactly as is
        text = re.sub(r'<\s*/\s*response\s*>', '</response>', text, flags=re.IGNORECASE)

        # replace '---' with <hr> (horizontal rule) for a separator line
        text = re.sub(r'---', '<hr>', text)

        # handle <think> tags
        open_pattern = r'<\s*think\s*>'
        close_pattern = r'<\s*/\s*think\s*>'

        # replace opening <think> tags
        text = re.sub(
            open_pattern, 
            '<span style="font-style: italic; font-weight: 200;">', 
            text,
            flags=re.IGNORECASE
        )

        # replace closing </think> tags
        text = re.sub(
            close_pattern,
            '</span><br><br>',
            text,
            flags=re.IGNORECASE
        )

        # clean up any partial/broken <think> tags that might remain
        # but avoid matching </div> and </response> in this cleanup
        partial_pattern = r'<\s*/?\s*th?i?n?k?\s*/?>'
        text = re.sub(partial_pattern, '', text, flags=re.IGNORECASE)

        return text


    def scroll_to_bottom(self):
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        self.scroll_area.setObjectName("scroll_area")

    def fade_in(self, widget):
        # create a QGraphicsOpacityEffect to animate the opacity
        self.effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(self.effect)
        # create a QPropertyAnimation to animate the opacity
        self.animation = QPropertyAnimation(self.effect, b"opacity")
        self.animation.setDuration(2000)   
        self.animation.setStartValue(0.0)   
        self.animation.setEndValue(1.0)   
        self.animation.start()

class OllamaWorker(QObject):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, user_message, conversation_history):
        super().__init__()
        self.user_message = user_message
        self.conversation_history = conversation_history.copy()
        self.num_threads = multiprocessing.cpu_count() / 2

    def run(self):
        try:
            stream = ollama.chat(
                model='deepseek-r1:8b',
                messages=self.conversation_history,
                stream=True,
                options = {
                    "num_thread": self.num_threads,
                    "temperature": 2.52,
                    "top_n": 121,
                    "f16_kv": True,
                    "num_ctx": 1024,
                    "num_batch": 8,
                    "num_prediction": 1024
                }
            )
            full_content = ''
            for chunk in stream:
                content = chunk['message']['content']
                print(content)
                full_content += content
                self.chunk_received.emit(content)
            
            self.finished.emit(full_content)

        except Exception as e:
            print(f"Error: {e}")
            error_message = "There seems to be a miscalculation."
            self.chunk_received.emit(error_message)
            self.finished.emit(error_message)