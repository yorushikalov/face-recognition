"""
preprocess.py — 数据预处理模块

功能：读取采集的人脸图像 → 灰度化+均衡化 → 统一尺寸 → 归一化 → 标签编码
输出：X_preprocessed.npy, labels.npy, label_encoder.pkl
"""
import os
import pickle
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_images(data_dir="data/facedb", img_size=(160, 160)):
    """
    读取 data/facedb/ 下每个子文件夹（每个人名）中的图片
    返回: images (列表), labels (列表)
    """
    images = []
    labels = []

    # 列出所有人物文件夹（如 BRQ/）
    person_dirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    if not person_dirs:
        print(f"❌ 在 {data_dir} 下未找到任何人脸数据文件夹")
        print("💡 请先运行 capture.py 采集照片")
        return None, None

    print(f"👤 检测到 {len(person_dirs)} 个人物: {person_dirs}")

    for person_name in person_dirs:
        person_dir = os.path.join(data_dir, person_name)
        files = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
        print(f"   📂 {person_name}: {len(files)} 张图片")

        for filename in files:
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. 灰度化
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. 直方图均衡化（增强对比度，抗光线干扰）
            equalized = cv2.equalizeHist(gray)

            # 3. 统一尺寸缩放
            resized = cv2.resize(equalized, img_size)

            images.append(resized)
            labels.append(person_name)

    return images, labels


def main():
    # 读取原始图片
    images, labels = load_images()
    if images is None or len(images) == 0:
        return

    # 转换为 numpy 数组并归一化
    X = np.array(images, dtype=np.float32) / 255.0       # 归一化 [0, 1]
    X = X.reshape(X.shape[0], 160, 160, 1)               # 添加通道维度 (160,160,1)

    # 标签编码：人名 → 数字
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\n📊 数据统计:")
    print(f"   ✅ 总样本数: {len(X)}")
    print(f"   📐 图片尺寸: {X.shape[1]}x{X.shape[2]}")
    print(f"   🏷️  类别数: {len(le.classes_)} → {list(le.classes_)}")
    print(f"   🔢 标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 划分训练集和测试集（80% 训练, 20% 测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 保存预处理后的数据
    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)

    with open("data/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"\n💾 数据保存完成:")
    print(f"   📁 data/X_train.npy     → {X_train.shape}")
    print(f"   📁 data/X_test.npy      → {X_test.shape}")
    print(f"   📁 data/y_train.npy     → {y_train.shape}")
    print(f"   📁 data/y_test.npy      → {y_test.shape}")
    print(f"   📁 data/label_encoder.pkl → 标签编码器")


if __name__ == "__main__":
    main()
