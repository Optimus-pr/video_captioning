from flask import Flask, render_template, request, jsonify
from predict_realtime import VideoDescriptionRealTime
import os
import config
import numpy as np

app = Flask(__name__)
video_to_text = VideoDescriptionRealTime(config)  # Make sure to instantiate with your actual config

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded video to a specific folder (e.g., 'uploads')
        print("hi",file.filename)
        video_path = os.path.join('static', 'uploads', file.filename)
        file.save(video_path)

        # Generate caption for the uploaded video
        video_to_text.test_path = video_path
      
        # file_name = file_list[self.num]
        path = os.path.join('data/testing_data/feat', file.filename + '.npy')
        
        f = np.load(path)
        sentence_predicted = video_to_text.greedy_search(f.reshape((-1, 80, 4096)))
        

        # Respond with caption and video path
        return jsonify({'caption': sentence_predicted, 'videoPath': f'uploads/{file.filename}'})

if __name__ == '__main__':
    app.run(debug=True)
