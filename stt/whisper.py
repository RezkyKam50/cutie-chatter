import torch
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration
)

class MicrophoneTranscriber:
    def __init__(self, model_name): 
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.config.forced_decoder_ids = None
        self.model.eval()

    def transcribe_audio(self, audio_data, sample_rate=16000):        
        input_features = self.processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features       
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        return transcription
    def record_and_transcribe(self, duration=5, sample_rate=16000):
        print(f"Recording for {duration} seconds...")
        
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()      
        audio_data = audio_data.flatten()
        transcription = self.transcribe_audio(audio_data, sample_rate)
        self.plot_audio(audio_data, sample_rate, transcription)
        print("Transcription:", transcription)
        return audio_data, transcription

    def plot_audio(self, audio_data, sample_rate, transcription):
        time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
        plt.figure(figsize=(10, 5))
        plt.plot(time, audio_data)
        plt.title('audio wave')
        plt.xlabel('waktu (detik)')
        plt.ylabel('amplitudo')
        plt.figtext(0.5, 0.01, f"Transkripsi: {transcription}", 
                    wrap=True, 
                    horizontalalignment='center', 
                    fontsize=12, 
                    color='blue',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        plt.tight_layout()
        plt.show()

# contoh penggunaan

if __name__ == "__main__":

    '''
    
        Selain whisper-small, bisa cek di web huggingface openai/whisper,
        ada versi tiny sampai large
    
    '''

    transcriber = MicrophoneTranscriber("openai/whisper-small") 
    while True:
        input("Tekan enter kemudian bicara 5 detik maks...")
        transcriber.record_and_transcribe()
