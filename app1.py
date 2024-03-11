from flask import Flask, render_template, Response
import cv2
import time

app = Flask(__name__)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    video_path = './data/testing_data/video/3qqEKTPxLNs_1_15.avi'  # Replace with the path to your video file
    
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
