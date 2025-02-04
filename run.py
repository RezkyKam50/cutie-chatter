import sys
import subprocess
from PyQt6.QtWidgets import QApplication
from cutie import CutieChatWindow


'''

    Make sure everything is thread safe

'''

def on_quit():
    subprocess.run(["ollama", "stop", "deepseek-r1:8b"], check=True)

def main_window():
    # create main instance on system level process
    app = QApplication(sys.argv)
    # connect thread to process for quit mechanism
    app.aboutToQuit.connect(on_quit)
    # deploy custom window instance
    window = CutieChatWindow()
    window.show()
    # event loop until exit on system call
    sys.exit(app.exec())

if __name__ == "__main__":
    main_window()
