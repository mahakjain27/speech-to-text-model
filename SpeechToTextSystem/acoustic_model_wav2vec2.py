from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

def transcribe_audio(file_path):
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    audio_input, rate = sf.read(file_path)
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

if __name__ == "__main__":
    sample_audio_path = "sample.wav"  # Replace with your audio file path
    transcription = transcribe_audio(sample_audio_path)
    print("Transcription:")
    print(transcription)
