from flask import Flask, request, render_template, redirect, url_for
import os
from acoustic_model_wav2vec2 import transcribe_audio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    transcription = None
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            transcription = transcribe_audio(file_path)
    return render_template('index.html', transcription=transcription)

if __name__ == '__main__':
    app.run(debug=True)
