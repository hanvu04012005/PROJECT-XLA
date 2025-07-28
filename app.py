from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import cv2
import io
import base64
import os
import logging
from werkzeug.utils import secure_filename

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Cấu hình upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}   
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load mô hình và dữ liệu với error handling
try:
    model = YOLO("best.pt")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

try:
    df_dinhduong = pd.read_excel("DinhDuong.xlsx")
    df_congdung = pd.read_excel("CongDung.xlsx")
    logger.info("Excel files loaded successfully")
except Exception as e:
    logger.error(f"Error loading Excel files: {e}")
    df_dinhduong = pd.DataFrame()
    df_congdung = pd.DataFrame()

# Lấy thông tin công dụng / dinh dưỡng
def lay_thong_tin_excel(ten, loai):
    ten = ten.strip().lower()
    if loai == "cong-dung":
        for _, row in df_congdung.iterrows():
            if row["Quả"].strip().lower() in ten:
                return row["Công dụng"]
    elif loai == "dinh-duong":
        for _, row in df_dinhduong.iterrows():
            if row["Quả"].strip().lower() in ten:
                return row["Dinh dưỡng"]
    return ""

# Hàm áp dụng chức năng ảnh
def xu_ly_anh(img_np, mode, sub_mode):
    if mode == "ro-net":
        pil_img = Image.fromarray(img_np)
        return np.array(pil_img.filter(ImageFilter.SHARPEN))

    elif mode == "sac-bien":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Tăng độ dày biên bằng morphological dilation
        kernel = np.ones((3, 3), np.uint8)
        edges_thick = cv2.dilate(edges, kernel, iterations=2)
        
        if sub_mode == "grayscale":
            return cv2.cvtColor(edges_thick, cv2.COLOR_GRAY2RGB)
        else:  # Giữ màu gốc, vẽ biên màu trắng
            img_out = img_np.copy()
            # Vẽ biên dày màu trắng lên ảnh gốc
            img_out[edges_thick > 0] = [255, 255, 255]
            return img_out

    return img_np

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Kiểm tra model đã load chưa
        if model is None:
            return jsonify({"error": "Model chưa được load. Vui lòng kiểm tra lại."}), 500

        files = request.files.getlist("images")
        
        # Validation files
        if not files or all(file.filename == '' for file in files):
            return jsonify({"error": "Vui lòng chọn ít nhất một file ảnh"}), 400
        
        # Kiểm tra file types
        for file in files:
            if not allowed_file(file.filename):
                return jsonify({"error": f"File {file.filename} không được hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        object_name = request.form.get("object_name", "").strip().lower()
        object_type = request.form.get("object_type", "").strip()
        function_mode = request.form.get("function_mode", "").strip()
        sub_mode = request.form.get("sub_mode", "giu-mau").strip()
        info_type = request.form.get("info_type", "").strip()

        # Validation input
        if object_type not in ["toan-anh", "doi-tuong"]:
            function_mode = ""
            sub_mode = ""

        result_images = []
        counts = {}

        for file in files:
            original_img = Image.open(file.stream).convert("RGB")
            img_np = np.array(original_img)

            # Giảm confidence threshold để nhận diện nhiều đối tượng hơn
            results = model.predict(original_img, conf=0.1)[0]
            names = model.names
            boxes = results.boxes

            if object_type == "toan-anh":
                img_draw = img_np.copy()
                detected_count = 0
                
                for box in boxes:
                    cls_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    name = names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Vẽ bounding box
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Vẽ nhãn với confidence
                    label = f"{name} ({confidence:.2f})"
                    cv2.putText(img_draw, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    counts[name] = counts.get(name, 0) + 1
                    detected_count += 1
                
                # Nếu không phát hiện được gì, thêm thông báo
                if detected_count == 0:
                    cv2.putText(img_draw, "Không phát hiện trái cây", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                img_processed = xu_ly_anh(img_draw, function_mode, sub_mode)
                img_pil = Image.fromarray(img_processed)

            elif object_type == "doi-tuong":
                found = False
                for box in boxes:
                    cls_id = int(box.cls.item())
                    name = names[cls_id].lower()
                    if object_name in name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = img_np[y1:y2, x1:x2]
                        img_processed = xu_ly_anh(crop, function_mode, sub_mode)
                        img_pil = Image.fromarray(img_processed)
                        counts[name] = counts.get(name, 0) + 1
                        found = True
                        break
                if not found:
                    img_pil = original_img

            else:
                # Nếu không chọn object_type, chỉ trả lại ảnh gốc
                img_pil = original_img

            # Encode base64
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            result_images.append("data:image/jpeg;base64," + img_str)

        # Trích xuất thông tin nếu có
        info = ""
        if object_type == "doi-tuong" and object_name and info_type:
            info = lay_thong_tin_excel(object_name, info_type)

        return jsonify({
            "images": result_images,
            "counts": counts,
            "info": info
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": "Có lỗi xảy ra khi xử lý ảnh. Vui lòng thử lại."}), 500

if __name__ == '__main__':
    app.run(debug=True)
