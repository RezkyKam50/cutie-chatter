import sys, subprocess
from PyQt6.QtWidgets import QApplication
from cutie import CutieTheCutest

'''

    Make sure everything is thread safe

'''

def on_quit():
    subprocess.run(["ollama", "stop", "deepseek-r1:8b"], check=True) 

if __name__ == "__main__":

    # deepseek-r1:1.5b, 7b, 8b, 14b, 32b, 627b
    model = "deepseek-r1:8b"
    # create main instance on system level process
    app = QApplication(sys.argv)
    # connect thread to process for quit mechanism
    app.aboutToQuit.connect(on_quit)
    # deploy custom window instance
    window = CutieTheCutest(model)
    window.show()
    # event loop until exit on system call
    sys.exit(app.exec())
