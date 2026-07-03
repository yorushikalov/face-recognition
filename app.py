"""
app.py — Flask Web 人脸识别应用

功能：提供 Web 界面的实时摄像头识别 + 上传照片识别
用法：python app.py
"""
import os
import pickle
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow import keras

# ---------- Flask 初始化 ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB 上传限制
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------- 加载模型（全局加载一次） ----------
print("🚀 加载模型中...")
feature_extractor = keras.models.load_model("models/feature_extractor.h5")

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("models/face_embeddings.pkl", "rb") as f:
    emb_data = pickle.load(f)
    centroids = emb_data["centroids"]
    thresholds = emb_data["thresholds"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print(f"✅ 模型加载完成，支持 {len(le.classes_)} 个人物: {list(le.classes_)}")

# 阈值调节系数：值越大越容易识别为"自己"，但误识别风险也会增加
# 默认 1.0，建议从 1.5 开始试，逐步增加至满意
THRESHOLD_MULTIPLIER = 1.8


# ---------- 核心函数 ----------
def preprocess_face(face_img, img_size=(160, 160)):
    """预处理人脸（与训练时完全一致）"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    resized = cv2.resize(equalized, img_size)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 160, 160, 1)


def recognize_face(face_img):
    """识别单张人脸，返回 (label, confidence, distance)"""
    processed = preprocess_face(face_img)
    embedding = feature_extractor.predict(processed, verbose=0)[0]

    best_label = "Unknown"
    best_distance = float("inf")
    best_class_idx = -1

    for class_idx, centroid in centroids.items():
        distance = np.linalg.norm(embedding - centroid)
        if distance < best_distance:
            best_distance = distance
            best_class_idx = class_idx

    effective_threshold = thresholds[best_class_idx] * THRESHOLD_MULTIPLIER
    if best_class_idx != -1 and best_distance < effective_threshold:
        best_label = le.inverse_transform([best_class_idx])[0]
        confidence = max(0, 1 - best_distance / effective_threshold)
    else:
        confidence = 0.0

    return best_label, confidence, best_distance


def detect_and_recognize(frame):
    """检测图片中所有人脸并识别"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    results = []
    for (x, y, w, h) in faces:
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        face_region = frame[y1:y2, x1:x2]

        label, confidence, distance = recognize_face(face_region)
        results.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "label": label,
            "confidence": round(float(confidence), 4),
            "distance": round(float(distance), 4),
        })

    return results


def draw_results(frame, results):
    """在图片上画框和标签"""
    for r in results:
        x, y, w, h = r["bbox"]
        is_known = r["label"] != "Unknown"
        color = (0, 255, 0) if is_known else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{r['label']} ({r['confidence']:.0%})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


# ---------- 路由 ----------
@app.route("/")
def index():
    """首页"""
    return render_template("index.html")


@app.route("/camera")
def camera():
    """摄像头实时识别页面"""
    return render_template("camera.html")


@app.route("/upload")
def upload():
    """上传照片识别页面"""
    return render_template("upload.html")


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """
    接收摄像头帧，进行人脸检测和识别
    输入: POST multipart, 字段名 'image', 值为 JPEG 图片
    输出: JSON { faces: [{bbox, label, confidence}, ...] }
    """
    if "image" not in request.files:
        return jsonify({"error": "没有图片数据"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "图片解码失败"}), 400

    results = detect_and_recognize(frame)
    return jsonify({"faces": results})


@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    """
    接收上传图片，返回标注图片 + JSON 结果
    输入: POST multipart, 字段名 'file', 值为图片文件
    输出: JSON { faces: [...], annotated_url: "...", success: true }
    """
    if "file" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "图片解码失败，请上传 JPG/PNG 格式"}), 400

    # 检测 + 识别
    results = detect_and_recognize(frame)
    annotated = draw_results(frame.copy(), results)

    # 保存标注后的图片
    filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    cv2.imwrite(save_path, annotated)

    return jsonify({
        "success": True,
        "faces": results,
        "annotated_url": url_for("static", filename=f"uploads/{filename}"),
        "img_width": frame.shape[1],
        "img_height": frame.shape[0],
    })


# ---------- 启动 ----------
if __name__ == "__main__":
    print("\n🌐 Flask Web 服务启动: http://127.0.0.1:5000")
    print("   📷 /camera  - 摄像头实时识别")
    print("   📤 /upload  - 上传照片识别")
    print("   ❌ Ctrl+C   - 停止服务\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
