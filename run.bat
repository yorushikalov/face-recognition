@echo off
chcp 65001 >nul
title 人脸识别系统

echo ========================================
echo   基于深度学习的 AI 人脸识别系统
echo   Flask + OpenCV + Keras + sklearn
echo ========================================
echo.

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] 首次使用，正在创建虚拟环境...
    python -m venv venv
    echo [INFO] 安装依赖...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo 启动方式：
echo   1. 样本采集  — 按 1
echo   2. 模型训练  — 按 2
echo   3. 命令行识别 — 按 3
echo   4. Web 服务   — 按 4
echo   5. 退出      — 按 5
echo.

set /p choice="请选择 (1-5): "

if "%choice%"=="1" (
    set /p name="输入你的姓名: "
    python capture.py --name %name% --count 200
    goto :end
)
if "%choice%"=="2" (
    python preprocess.py
    python train.py
    goto :end
)
if "%choice%"=="3" (
    python recognize.py
    goto :end
)
if "%choice%"=="4" (
    echo.
    echo 🌐 浏览器访问: http://127.0.0.1:5000
    echo.
    python app.py
    goto :end
)
if "%choice%"=="5" goto :end

echo 无效输入！
:end
pause
