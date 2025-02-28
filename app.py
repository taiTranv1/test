from flask import Flask, render_template
from flask_socketio import SocketIO, emit
#form werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

#khai bao' flask
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")
model = YOLO("best.pt")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    img_data = base64.b64decode(data.split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame) #chay yolo
    results_frame = results[0].plot()

    #ma hoa' kq gui ve client
    _, buffer = cv2.imencode(".jpg", results_frame)
    results_data = base64.b64encode(buffer).decode("utf-8")
    emit("result","data:image/jpeg;base64," + results_data)

if __name__ == "__main__" :
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=5000, debug=True) 