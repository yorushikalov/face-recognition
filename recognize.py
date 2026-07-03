"""
recognize.py — 命令行实时人脸识别脚本

功能：打开摄像头 → 检测人脸 → 用训练好的模型识别身份 → 实时显示结果
用法：python recognize.py
"""
import os
import pickle
import cv2
import numpy as np
from tensorflow import keras


def load_models():
    """加载训练好的模型和特征中心"""
    feature_extractor = keras.models.load_model("models/feature_extractor.h5")

    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    with open("models/face_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        centroids = data["centroids"]
        thresholds = data["thresholds"]

    return feature_extractor, le, centroids, thresholds


# 阈值调节系数（和 app.py 保持一致）
THRESHOLD_MULTIPLIER = 1.8


def preprocess_face(face_img, img_size=(160, 160)):
    """对检测到的人脸进行预处理，保持和训练时一致"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    resized = cv2.resize(equalized, img_size)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 160, 160, 1)


def recognize_face(feature_extractor, le, centroids, thresholds, face_img):
    """
    识别单张人脸图片
    返回: (预测人名, 置信度/距离, 是否识别成功)
    """
    processed = preprocess_face(face_img)
    embedding = feature_extractor.predict(processed, verbose=0)[0]

    best_label = "Unknown"
    best_distance = float("inf")
    best_class_idx = -1

    # 遍历每个类别的人脸中心，找距离最近的
    for class_idx, centroid in centroids.items():
        distance = np.linalg.norm(embedding - centroid)
        if distance < best_distance:
            best_distance = distance
            best_class_idx = class_idx

    # 如果最近距离小于阈值，则识别成功
    effective_threshold = thresholds[best_class_idx] * THRESHOLD_MULTIPLIER
    if best_class_idx != -1 and best_distance < effective_threshold:
        best_label = le.inverse_transform([best_class_idx])[0]
        confidence = max(0, 1 - best_distance / effective_threshold)
    else:
        confidence = 0.0

    return best_label, best_distance, confidence


def main():
    print("🚀 加载模型...")
    feature_extractor, le, centroids, thresholds = load_models()
    print(f"✅ 模型加载完成，支持 {len(le.classes_)} 个人物: {list(le.classes_)}")

    # 加载 Haar 人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("\n💡 按 [q] 或 [ESC] 退出\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
        )

        for (x, y, w, h) in faces:
            # 裁剪人脸区域
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            face_region = frame[y1:y2, x1:x2]

            # 识别
            label, distance, confidence = recognize_face(
                feature_extractor, le, centroids, thresholds, face_region
            )

            # 根据识别结果选择颜色
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

            # 画框和标签
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label} ({confidence:.2%})"
            cv2.putText(
                frame, text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2,
            )

        cv2.imshow("Real-time Face Recognition - Press [Q/ESC] to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 识别结束")


if __name__ == "__main__":
    main()
