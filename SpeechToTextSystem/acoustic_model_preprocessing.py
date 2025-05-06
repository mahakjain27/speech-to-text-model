import librosa
import librosa.display
import matplotlib.pyplot as plt

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfcc, sr

def plot_mfcc(mfcc):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample_audio_path = "sample.wav"  # Replace with your audio file path
    mfcc, sr = preprocess_audio(sample_audio_path)
    plot_mfcc(mfcc)
