import pytesseract
from pdf2image import convert_from_path
from PyQt6.QtWidgets import QApplication, QFileDialog, QWidget


class TextExtractor:
    """Handles the extraction of text from both images and PDFs."""
    
    def __init__(self):
        pass
    
    def extract_text_from_image(self, image):
        """Extracts text from an image using pytesseract."""
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error reading image: {e}"

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from all pages of a PDF."""
        try:
            images = convert_from_path(pdf_path)
            full_text = ""
            for i, image in enumerate(images):
                print(f"Processing page {i+1}...")
                text = self.extract_text_from_image(image)
                full_text += f"--- Page {i+1} ---\n{text}\n\n"
            return full_text
        except Exception as e:
            return f"Error processing PDF: {e}"


class FileSelector(QWidget):
    """Handles the file selection dialog."""
    
    def __init__(self):
        super().__init__()

    def open_file_dialog(self):
        """Opens a file dialog to select a PDF file."""
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF File", "", "PDF Files (*.pdf);;All Files (*)", options=options
        )
        return file_path
