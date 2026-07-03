@echo off
title AI Face Recognition

:: Check venv
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    echo [SETUP] Installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:menu
echo.
echo ========================================
echo   AI Face Recognition System
echo ========================================
echo   1. Capture faces
echo   2. Train model
echo   3. CLI recognition
echo   4. Start Web server
echo   5. Exit
echo ========================================
set /p choice="Select (1-5): "

if "%choice%"=="1" (
    set /p name="Enter your name: "
    python capture.py --name %name% --count 200
    goto end
)
if "%choice%"=="2" (
    python preprocess.py
    python train.py
    goto end
)
if "%choice%"=="3" (
    python recognize.py
    goto end
)
if "%choice%"=="4" (
    echo.
    echo Open browser: http://127.0.0.1:5000
    echo.
    python app.py
    goto end
)
if "%choice%"=="5" goto end

echo Invalid input!
goto menu

:end
pause
