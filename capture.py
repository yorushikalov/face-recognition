"""
capture.py — 人脸样本采集模块

功能：打开摄像头 → 检测人脸 → 保存人脸图像到 data/facedb/<姓名>/ 目录
用法：python capture.py --name 你的姓名 --count 100
"""
import argparse
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="人脸样本采集工具")
    parser.add_argument("--name", type=str, required=True, help="你的姓名（将作为文件夹名）")
    parser.add_argument("--count", type=int, default=100, help="采集张数（默认100）")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name
    max_count = args.count

    # 创建保存目录：data/facedb/<姓名>/
    save_dir = os.path.join("data", "facedb", name)
    os.makedirs(save_dir, exist_ok=True)

    # 加载 OpenCV 内置的 Haar Cascade 人脸检测模型
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 打开摄像头（0 = 默认第一个摄像头）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查是否被其他应用占用")
        return

    saved_count = 0
    frame_skip = 0  # 简单的跳帧计数器，避免保存太相似的照片

    print(f"\n{'='*50}")
    print(f"🎯 开始采集 【{name}】 的人脸样本")
    print(f"📸 目标张数: {max_count}")
    print(f"{'='*50}")
    print("💡 操作提示:")
    print("   [空格键] 或 [s]   → 手动保存当前检测到的人脸")
    print("   人脸对准摄像头，自动保存...")
    print("   [q] 或 [ESC]      → 退出\n")

    while saved_count < max_count:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取摄像头帧失败")
            break

        # 转灰度（Haar 检测需要灰度图）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 执行人脸检测
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # 每次缩小比例，值越小检测越慢但越准
            minNeighbors=5,       # 每个候选框需要至少5个邻居确认
            minSize=(100, 100),   # 最小人脸尺寸，过滤掉太小的误检
        )

        # 在画面中标记检测到的人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Detected: {len(faces)}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # 如果检测到恰好一张人脸 → 自动保存（每5帧保存一次，避免重复）
        if len(faces) == 1 and frame_skip % 5 == 0:
            (x, y, w, h) = faces[0]
            # 裁剪人脸区域（稍微扩展一点边距）
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            face_region = frame[y1:y2, x1:x2]

            # 保存人脸图像
            filename = os.path.join(save_dir, f"img_{saved_count:04d}.jpg")
            cv2.imwrite(filename, face_region)
            saved_count += 1

            # 在画面上显示保存进度
            cv2.putText(
                frame,
                f"✅ Saved: {saved_count}/{max_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        frame_skip += 1

        # 显示实时画面
        cv2.imshow("Face Capture - Press [Space/S] to save, [Q/ESC] to quit", frame)

        # 键盘事件处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q 或 ESC 退出
            print("\n👋 用户主动退出")
            break
        elif key == ord(" ") or key == ord("s"):  # 空格 或 s 手动保存
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                face_region = frame[y1:y2, x1:x2]
                filename = os.path.join(save_dir, f"manual_{saved_count:04d}.jpg")
                cv2.imwrite(filename, face_region)
                saved_count += 1
                print(f"📸 手动保存第 {saved_count}/{max_count} 张")
            else:
                print("⚠️ 未检测到人脸，无法保存")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 统计结果
    actual_files = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    print(f"\n{'='*50}")
    print(f"✅ 采集完成！共保存 {actual_files} 张人脸图像")
    print(f"📂 保存位置: {save_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
