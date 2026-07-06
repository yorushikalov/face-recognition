"""
app.py — Flask Web 人脸识别应用

功能：用户注册/登录（密码/短信/人脸）、摄像头识别、上传照片识别
用法：python app.py
"""
import os
import pickle
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, session, redirect
from functools import wraps
from tensorflow import keras
from database import init_db, create_user, verify_password, verify_phone
from database import get_user_by_id, get_all_face_users, generate_sms_code, verify_sms_code
from database import update_face_embedding
from notify import user_registered, user_logged_in, user_logged_out

# ---------- Flask 初始化 ----------
app = Flask(__name__)
app.secret_key = "face-recognition-secret-key-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB 上传限制
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 初始化数据库
init_db()

# 注册过程中的人脸特征临时存储（不用 session cookie，避免 4KB 限制）
_reg_face_data = {}  # {user_id: {"embeddings": [np.array, ...], "count": int}}

# ---------- 加载模型（全局加载一次） ----------
print("[INFO] Loading models...")
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
print(f"[OK] Models loaded, {len(le.classes_)} person(s): {list(le.classes_)}")

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


# ---------- 登录验证装饰器 ----------
def login_required(f):
    """要求登录后才能访问"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def extract_face_embedding(face_img):
    """提取单张人脸图片的 128 维特征向量"""
    processed = preprocess_face(face_img)
    embedding = feature_extractor.predict(processed, verbose=0)[0]
    return embedding


def match_face(embedding):
    """
    在数据库中匹配人脸，返回 (user_dict, distance) 或 (None, inf)
    """
    users = get_all_face_users()
    if not users:
        return None, float("inf")

    best_user = None
    best_distance = float("inf")

    for user in users:
        if user["face_embedding"] is None:
            continue
        distance = np.linalg.norm(embedding - user["face_embedding"])
        if distance < best_distance:
            best_distance = distance
            best_user = user

    return best_user, best_distance


# ---------- 路由 ----------
# === 页面路由 ===
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


# === 认证页面路由 ===
@app.route("/login")
def login_page():
    """登录页面"""
    return render_template("login.html")


@app.route("/register")
def register_page():
    """注册页面"""
    return render_template("register.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """用户仪表盘"""
    user = get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login_page"))
    return render_template("dashboard.html", user=user)


@app.route("/logout")
def logout():
    """退出登录"""
    user = get_user_by_id(session.get("user_id"))
    real_name = user["real_name"] if user else session.get("username", "未知用户")
    session.clear()
    user_logged_out(real_name)
    return redirect(url_for("login_page"))


# === 认证 API 路由 ===
@app.route("/api/login_password", methods=["POST"])
def api_login_password():
    """账号密码登录"""
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "message": "用户名和密码不能为空"})

    success, result = verify_password(username, password)
    if success:
        session["user_id"] = result["id"]
        session["username"] = result["username"]
        user_logged_in(result["real_name"], "密码")
        return jsonify({"success": True, "message": "登录成功", "user": _safe_user(result)})

    return jsonify({"success": False, "message": result})


@app.route("/api/send_sms", methods=["POST"])
def api_send_sms():
    """发送短信验证码（模拟）"""
    data = request.get_json()
    phone = data.get("phone", "").strip()

    if not phone or len(phone) != 11 or not phone.isdigit():
        return jsonify({"success": False, "message": "请输入正确的 11 位手机号"})

    # 检查手机号是否已注册
    exists, _ = verify_phone(phone)
    if not exists:
        return jsonify({"success": False, "message": "该手机号未注册"})

    code = generate_sms_code(phone)
    print(f"\n[SMS Demo] Phone {phone}, Code: {code}\n")
    return jsonify({"success": True, "message": f"验证码已发送（演示模式: {code}）", "demo_code": code})


@app.route("/api/login_sms", methods=["POST"])
def api_login_sms():
    """手机验证码登录"""
    data = request.get_json()
    phone = data.get("phone", "").strip()
    code = data.get("code", "").strip()

    if not phone or not code:
        return jsonify({"success": False, "message": "手机号和验证码不能为空"})

    success, msg = verify_sms_code(phone, code)
    if not success:
        return jsonify({"success": False, "message": msg})

    # 验证码正确，获取用户
    _, user = verify_phone(phone)
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    user_logged_in(user["real_name"], "短信验证码")
    return jsonify({"success": True, "message": "登录成功", "user": _safe_user(user)})


@app.route("/api/login_face", methods=["POST"])
def api_login_face():
    """人脸识别登录"""
    if "image" not in request.files:
        return jsonify({"success": False, "message": "没有图片数据"})

    file = request.files["image"]
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"success": False, "message": "图片解码失败"})

    # 检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    if len(faces) == 0:
        return jsonify({"success": False, "message": "未检测到人脸，请正对摄像头"})

    if len(faces) > 1:
        return jsonify({"success": False, "message": "检测到多张人脸，请确保只有您一人在镜头中"})

    # 提取人脸并识别
    x, y, w, h = faces[0]
    face_region = frame[y:y + h, x:x + w]
    embedding = extract_face_embedding(face_region)

    user, distance = match_face(embedding)

    # 人脸匹配阈值（和训练阈值类似的逻辑）
    FACE_LOGIN_THRESHOLD = 2.5  # 可根据实际效果调整

    if user is None or distance > FACE_LOGIN_THRESHOLD:
        return jsonify({
            "success": False,
            "message": f"人脸识别失败，未找到匹配用户（距离: {distance:.4f}）",
            "distance": round(float(distance), 4),
        })

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    user_logged_in(user["real_name"], "人脸识别")
    return jsonify({
        "success": True,
        "message": f"人脸识别成功，欢迎 {user['real_name']}",
        "user": _safe_user(user),
        "distance": round(float(distance), 4),
    })


@app.route("/api/register", methods=["POST"])
def api_register():
    """账号注册（不含人脸，人脸单独上传）"""
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    phone = data.get("phone", "").strip()
    real_name = data.get("real_name", "").strip()

    # 验证
    if not all([username, password, phone, real_name]):
        return jsonify({"success": False, "message": "请填写所有字段"})

    if len(username) < 3:
        return jsonify({"success": False, "message": "用户名至少 3 个字符"})

    if len(password) < 6:
        return jsonify({"success": False, "message": "密码至少 6 个字符"})

    if len(phone) != 11 or not phone.isdigit():
        return jsonify({"success": False, "message": "请输入正确的 11 位手机号"})

    success, msg = create_user(username, password, phone, real_name)
    if success:
        # 获取刚创建的用户 ID
        _, user = verify_password(username, password)
        session["_reg_user_id"] = user["id"]  # 暂存用于后续人脸录入
        return jsonify({"success": True, "message": "注册成功，请录入人脸数据", "user_id": user["id"]})

    return jsonify({"success": False, "message": msg})


@app.route("/api/register_face", methods=["POST"])
def api_register_face():
    """注册时上传人脸照片"""
    user_id = session.get("_reg_user_id")
    if not user_id:
        return jsonify({"success": False, "message": "请先完成账号注册"})

    if "image" not in request.files:
        return jsonify({"success": False, "message": "没有图片数据"})

    file = request.files["image"]
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"success": False, "message": "图片解码失败"})

    # 检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    if len(faces) == 0:
        return jsonify({"success": False, "message": "未检测到人脸，请正对摄像头"})

    if len(faces) > 1:
        return jsonify({"success": False, "message": "检测到多张人脸"})

    # 保存人脸照片
    user_dir = f"data/facedb/user_{user_id}"
    os.makedirs(user_dir, exist_ok=True)

    x, y, w, h = faces[0]
    face_region = frame[y:y + h, x:x + w]
    file_count = len(os.listdir(user_dir))
    cv2.imwrite(f"{user_dir}/face_{file_count:04d}.jpg", face_region)

    # 提取特征
    embedding = extract_face_embedding(face_region)

    # 累积计算平均特征向量（存内存字典，避开 session cookie 4KB 限制）
    if user_id not in _reg_face_data:
        _reg_face_data[user_id] = {"embeddings": [], "count": 0}
    _reg_face_data[user_id]["embeddings"].append(embedding)
    _reg_face_data[user_id]["count"] += 1

    face_count = _reg_face_data[user_id]["count"]

    return jsonify({
        "success": True,
        "message": f"已采集 {face_count} 张人脸",
        "face_count": face_count,
        "bbox": [int(x), int(y), int(w), int(h)],
    })


@app.route("/api/register_finish", methods=["POST"])
def api_register_finish():
    """完成注册，保存最终人脸特征"""
    user_id = session.get("_reg_user_id")
    if not user_id:
        return jsonify({"success": False, "message": "注册流程异常"})

    user_data = _reg_face_data.get(user_id, {})
    embeddings = user_data.get("embeddings", [])
    if len(embeddings) < 3:
        return jsonify({"success": False, "message": f"请至少采集 3 张人脸照片（当前 {len(embeddings)} 张）"})

    avg_embedding = np.mean(embeddings, axis=0)
    update_face_embedding(user_id, avg_embedding, len(embeddings))

    # 清理临时数据
    _reg_face_data.pop(user_id, None)
    session.pop("_reg_user_id", None)

    # 获取用户名发送通知
    user = get_user_by_id(user_id)
    if user:
        user_registered(user["real_name"])

    return jsonify({"success": True, "message": "注册完成！人脸数据已保存"})


def _safe_user(user):
    """返回安全的用户信息（不含密码哈希和二进制数据）"""
    return {
        "id": user["id"],
        "username": user["username"],
        "phone": user["phone"],
        "real_name": user["real_name"],
        "has_face": user["face_embedding"] is not None,
    }


# ---------- 启动 ----------
if __name__ == "__main__":
    print("\n[Flask] Server running at: http://127.0.0.1:5000")
    print("   /login     - User login")
    print("   /register  - Register")
    print("   /camera    - Camera recognition")
    print("   /upload    - Upload recognition")
    print("   Ctrl+C     - Stop server\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
