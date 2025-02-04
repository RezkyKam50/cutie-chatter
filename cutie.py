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
from backends import ChatWidget, OllamaWorker

import os, screeninfo

class CutieChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CutieChatter")

        self.screen_dimension = screeninfo.get_monitors()
        self.last_screen_dimension = self.screen_dimension[-1]
        WIN_W, WIN_H = self.last_screen_dimension.width, self.last_screen_dimension.height
        self.setGeometry(100, 100, int(WIN_W / 2), int(WIN_H / 2.5))
        icon_dir = os.path.dirname(os.path.abspath(__file__))
        self.chat_widget = ChatWidget()

        # title bar setup
        title_bar = QWidget(self)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title_layout.addSpacerItem(spacer)

        self.minimize_button = QPushButton("minimize", self)
        self.exit_button = QPushButton("exit", self)
        self.minimize_button.setObjectName("minimize_button")
        self.exit_button.setObjectName("exit_button")
        self.minimize_button.clicked.connect(self.minimize_window)
        self.exit_button.clicked.connect(self.close_window)

        title_layout.addWidget(self.minimize_button)
        title_layout.addWidget(self.exit_button)
        title_bar.setMinimumHeight(60)

        central_widget = QWidget(self)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(title_bar, alignment=Qt.AlignmentFlag.AlignTop)

        # directly add chat_widget with no spacer between it and the prompt_box
        main_layout.addWidget(self.chat_widget)

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(10, 10, 10, 10)

        self.prompt_box = QTextEdit(self)
        self.prompt_box.setPlaceholderText("Message DeepSeek")
        self.prompt_box.setObjectName("prompt_box")
        self.prompt_box.setMinimumHeight(int(WIN_H * 0.06))
        self.prompt_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chat_widget.setMinimumHeight(int(WIN_H / 1.8))

        self.submit_button = QPushButton(self)
        self.submit_button.setObjectName("submit_button")
        icon_send_path = os.path.join(icon_dir, "icons", "send.svg")
        self.submit_button.setIcon(QIcon(icon_send_path))
        self.submit_button.setIconSize(QSize(34, 34))
        self.submit_button.clicked.connect(self.on_submit)

        input_layout.addWidget(self.prompt_box)
        input_layout.addWidget(self.submit_button)

        main_layout.addLayout(input_layout)

        self.setCentralWidget(central_widget)
        self.apply_theme()


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


    def minimize_window(self):
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMinimized)

    def close_window(self):
        self.close()

    def on_submit(self):
        entered_text = self.prompt_box.toPlainText().strip()
        if entered_text:
            self.animate_submit_button()

            self.chat_widget.add_user_message(entered_text)
            self.prompt_box.clear()
            self.chat_widget.start_ai_message()

            # add the user message to the conversation history
            self.conversation_history.append({'role': 'user', 'content': entered_text})

            self.run_ollama_chat()

    def run_ollama_chat(self):
        if self.is_ollama_thread_running:
            return

        self.is_ollama_thread_running = True
        self.submit_button.setEnabled(False)

        self.ollama_thread = QThread()
        self.ollama_worker = OllamaWorker(self.conversation_history[-1]['content'], self.conversation_history)
        self.ollama_worker.moveToThread(self.ollama_thread)

        self.ollama_worker.chunk_received.connect(self.chat_widget.append_to_ai_message)
        self.ollama_worker.finished.connect(self.on_ollama_response_complete)
        self.ollama_thread.started.connect(self.ollama_worker.run)
        self.ollama_thread.start()

    def on_ollama_response_complete(self, response):
        # add the AI's response to the conversation history
        self.conversation_history.append({'role': 'assistant', 'content': response})
        self.is_ollama_thread_running = False
        self.submit_button.setEnabled(True)
        self.ollama_thread.quit()
        self.ollama_worker.deleteLater()

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
        self.animation.setDuration(1000)

        self.animation.setEasingCurve(QEasingCurve.Type.CosineCurve)

        self.animation.setStartValue(QSize(34, 34))
        self.animation.setEndValue(QSize(26, 26))

        self.animation.start()
        self.animation.finished.connect(self.reset_submit_button_animation)

    def reset_submit_button_animation(self):
        # Reset the icon size to its original size
        self.submit_button.setIconSize(QSize(34, 34))

 
 


 

 
