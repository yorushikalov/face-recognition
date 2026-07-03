"""
train.py — 模型训练模块（带数据增强版）

功能：用 Keras CNN 提取人脸特征，计算每个人脸的"特征中心"和距离阈值
      加入数据增强提升单样本场景的泛化能力
输出：
    models/face_model.h5          - 完整 CNN 分类模型
    models/feature_extractor.h5   - 仅用于提取 128 维特征向量的模型
    models/face_embeddings.pkl    - 每个人的平均特征向量 + 识别阈值
    models/label_encoder.pkl       - 标签编码器
    models/history.png            - 训练损失曲线图
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data():
    """加载预处理后的数据"""
    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")
    return X_train, X_test, y_train, y_test


def build_cnn(input_shape, num_classes):
    """构建 CNN 分类模型（加入 BatchNormalization 更稳定）"""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),

        Dense(128, activation="relu", name="embedding"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_data_augmentor():
    """
    数据增强生成器 — 每张原图在每轮训练时都被随机变换
    相当于把 200 张图变成无穷多种变体
    """
    return ImageDataGenerator(
        rotation_range=20,         # 随机旋转 ±20°
        width_shift_range=0.1,     # 水平随机偏移 10%
        height_shift_range=0.1,    # 垂直随机偏移 10%
        brightness_range=(0.8, 1.2), # 随机调整亮度
        zoom_range=0.15,           # 随机缩放 ±15%
        horizontal_flip=True,      # 水平翻转
        fill_mode="nearest",       # 填充模式（防止旋转后黑边）
    )


def extract_embeddings(model, X):
    """用训练好的 CNN 提取 embedding 层输出"""
    feature_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output,
    )
    embeddings = feature_extractor.predict(X, batch_size=8, verbose=0)
    return embeddings, feature_extractor


def compute_centroids_and_thresholds(embeddings, y, labels):
    """
    计算每个类别的"特征中心"和识别阈值
    使用 95% 分位数代替 mean+std，对极端值更鲁棒
    """
    centroids = {}
    thresholds = {}

    for class_idx in range(len(labels)):
        class_embeddings = embeddings[y == class_idx]
        centroid = np.mean(class_embeddings, axis=0)
        centroids[class_idx] = centroid

        distances = np.linalg.norm(class_embeddings - centroid, axis=1)

        # 使用 95% 分位数 + 10% 余量（比 mean+std 更宽松，容错更好）
        threshold = np.percentile(distances, 95) * 1.1
        thresholds[class_idx] = threshold

        print(f"   🏷️  {labels[class_idx]}: {len(class_embeddings)} 样本, "
              f"平均距离={np.mean(distances):.4f}, 阈值={threshold:.4f}")

    return centroids, thresholds


def plot_history(history, save_path="models/history.png"):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    print("🚀 开始训练人脸识别模型（数据增强版）\n")

    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_data()
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    print(f"📊 训练数据: {X_train.shape}, 测试数据: {X_test.shape}, 类别数: {num_classes}")

    # 2. 构建 CNN
    model = build_cnn(input_shape, num_classes)
    model.summary()

    # 3. 数据增强生成器
    datagen = get_data_augmentor()
    # 对训练数据采用数据增强，测试数据不做增强
    train_flow = datagen.flow(X_train, y_train, batch_size=16)

    # 4. 训练 CNN（数据增强）
    print("\n⏳ 开始训练 CNN（每张图都会随机变换）...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, verbose=1, min_lr=1e-6),
    ]

    history = model.fit(
        train_flow,
        validation_data=(X_test, y_test),
        epochs=100,
        steps_per_epoch=max(1, len(X_train) // 16),
        callbacks=callbacks,
        verbose=1,
    )

    # 5. 评估模型
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 测试集准确率: {accuracy:.4f}")

    # 6. 提取特征向量
    print("\n🔍 提取人脸特征向量...")
    embeddings, feature_extractor = extract_embeddings(model, X_train)

    # 7. 计算每个类别的特征中心和阈值
    print("\n📐 计算识别阈值...")
    from sklearn.preprocessing import LabelEncoder
    with open("data/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    labels = le.classes_

    centroids, thresholds = compute_centroids_and_thresholds(embeddings, y_train, labels)

    # 8. 保存模型和数据
    os.makedirs("models", exist_ok=True)
    model.save("models/face_model.h5")
    feature_extractor.save("models/feature_extractor.h5")

    with open("models/face_embeddings.pkl", "wb") as f:
        pickle.dump({"centroids": centroids, "thresholds": thresholds}, f)

    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    plot_history(history)

    print("\n✅ 训练完成！保存文件如下:")
    print("   📁 models/face_model.h5")
    print("   📁 models/feature_extractor.h5")
    print("   📁 models/face_embeddings.pkl")
    print("   📁 models/label_encoder.pkl")
    print("   📁 models/history.png")


if __name__ == "__main__":
    main()
