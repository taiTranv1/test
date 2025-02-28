from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

# Khởi tạo Flask và SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

# Load YOLO model
try:
    model = YOLO("best.pt")
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLO: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    try:
        # Kiểm tra dữ liệu đầu vào
        if not data or "," not in data:
            emit("error", "Dữ liệu hình ảnh không hợp lệ")
            return

        # Giải mã hình ảnh từ base64
        img_data = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Kiểm tra xem frame có hợp lệ không
        if frame is None:
            emit("error", "Không thể giải mã hình ảnh")
            return

        # Xử lý hình ảnh bằng YOLO
        if model is not None:
            results = model(frame)  # Chạy YOLO
            results_frame = results[0].plot()

            # Mã hóa kết quả và gửi về client
            _, buffer = cv2.imencode(".jpg", results_frame)
            results_data = base64.b64encode(buffer).decode("utf-8")
            emit("result", "data:image/jpeg;base64," + results_data)
        else:
            emit("error", "Mô hình YOLO chưa được tải")
    except Exception as e:
        print(f"Lỗi khi xử lý frame: {e}")
        emit("error", "Đã xảy ra lỗi khi xử lý hình ảnh")

if __name__ == "__main__":
    # Lấy cổng từ biến môi trường hoặc mặc định là 5000
    port = int(os.environ.get("PORT", 5000))
    
    # Chạy ứng dụng
    socketio.run(app, host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "False").lower() == "true")