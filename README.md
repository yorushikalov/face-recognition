# 基于深度学习的 AI 人脸识别系统

> 参考陈强《Python项目开发实战》第5章实现  
> 技术栈：Flask + OpenCV-Python + Keras + scikit-learn

## 功能

- 📷 **摄像头实时识别**：命令行或浏览器打开摄像头，实时检测并识别人脸
- 📤 **上传照片识别**：上传照片，系统自动检测人脸并标注身份
- 🧠 **深度学习模型**：Keras CNN 提取人脸特征 + 距离阈值分类

## 快速开始

### 1. 环境要求

- Python 3.10.x
- 摄像头（可选，仅摄像头识别需要）
- 8GB+ 内存（训练时）

### 2. 安装

```bash
# 创建虚拟环境
python -m venv venv

# 激活（Windows）
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 使用流程

**① 采集人脸样本**
```bash
python capture.py --name 你的姓名 --count 200
```
> 尽量多角度拍摄（正脸、左右侧、抬头、低头、表情变化）

**② 数据预处理**
```bash
python preprocess.py
```

**③ 训练模型**
```bash
python train.py
```

**④ 命令行实时识别**
```bash
python recognize.py
```

**⑤ Web 服务**
```bash
python app.py
```
浏览器访问 http://127.0.0.1:5000

### 一键启动

Windows 下直接双击 `run.bat`，按菜单选择操作。

## 项目结构

```
face-recognition/
├── capture.py              # 样本采集
├── preprocess.py           # 数据预处理
├── train.py                # 模型训练（数据增强）
├── recognize.py            # 命令行实时识别
├── app.py                  # Flask Web 应用
├── templates/              # HTML 模板
├── static/                 # 静态资源 (CSS/JS)
├── data/facedb/            # 采集的人脸图片
├── models/                 # 训练好的模型文件
└── requirements.txt        # 依赖清单
```

## 调参说明

如果识别准确率不理想，调整 `app.py` 中的阈值系数：

```python
THRESHOLD_MULTIPLIER = 1.8  # 越大越容易识别，越小越严格
```

## 注意事项

- 训练过程使用 CPU，无需 GPU
- 采集照片时确保光线充足、面部清晰
- 单用户识别效果佳，多用户需重新采集数据并训练

## 微信通知（可选）

系统支持通过 PushPlus 向微信推送**注册/登录/登出**通知：

1. 访问 [pushplus.plus](https://www.pushplus.plus)，微信扫码登录
2. 点「发送消息」→「一对多消息」，复制你的 Token
3. 将 `notify.py.example` 重命名为 `notify.py`，填入 Token：
