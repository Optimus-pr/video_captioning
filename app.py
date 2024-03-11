from flask import Flask, render_template, request, jsonify,Response
from predict_realtime import VideoDescriptionRealTime
import os
import time
import cv2
import config
import numpy as np

app = Flask(__name__)
video_to_text = VideoDescriptionRealTime(config)  # Make sure to instantiate with your actual config

@app.route('/')
def index():
    return render_template('index.html')

uploaded_file_path = "./data/testing_data/video/77iDIp40m9E_126_131.avi"

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    video_path = uploaded_file_path  # Replace with the path to your video file
    
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Adjust the delay based on the desired playback speed
        time.sleep(0.1 / 2)

@app.route('/api/upload', methods=['POST'])
def upload_file():

    global uploaded_file_path

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
        
        uploaded_file_path = f'static/uploads/{file.filename}'

        # Respond with caption and video path
        return jsonify({'caption': sentence_predicted, 'videoPath': f'uploads/{file.filename}'})

if __name__ == '__main__':
    app.run(debug=True)
