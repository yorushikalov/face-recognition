/**
 * camera.js — 浏览器摄像头识别前端逻辑
 *
 * 流程：
 * 1. getUserMedia 获取摄像头流
 * 2. 每隔约 500ms 截取一帧画面发送到 Flask 后端
 * 3. 在后端完成人脸检测+识别
 * 4. 在前端 Canvas 上绘制检测框和标签
 */

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const resultText = document.getElementById('result-text');
const fpsDisplay = document.getElementById('fps-display');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');

let stream = null;
let isRunning = false;
let lastFrameTime = 0;
const FRAME_INTERVAL = 500; // ms，每 0.5 秒发一帧

/**
 * 异步截取 video 当前帧，转为 Blob 发送至 Flask 后端
 */
async function sendFrame() {
    if (!isRunning) return;

    // 将 video 当前帧画到 canvas 上
    ctx.drawImage(video, 0, 0, overlay.width, overlay.height);

    // canvas → Blob (JPEG, 质量 0.8 以平衡速度与清晰度)
    const blob = await new Promise(resolve => overlay.toBlob(resolve, 'image/jpeg', 0.8));

    // 构造 FormData 并发送
    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');

    try {
        const resp = await fetch('/predict_frame', { method: 'POST', body: formData });
        const data = await resp.json();
        handleResult(data);
    } catch (err) {
        console.error('请求失败:', err);
    }

    // 清空 canvas（避免残留），保留视频背景
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

/**
 * 处理后端返回的识别结果
 * data = { faces: [{ bbox: [x,y,w,h], label, confidence }] }
 */
function handleResult(data) {
    if (!data.faces || data.faces.length === 0) {
        resultText.textContent = '未检测到人脸';
        resultText.style.color = '#5f6368';
        return;
    }

    const face = data.faces[0];
    const [x, y, w, h] = face.bbox;

    // 在 canvas 上画框和标签
    const isKnown = face.label !== 'Unknown';
    ctx.strokeStyle = isKnown ? '#34a853' : '#ea4335';
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = isKnown ? '#34a853' : '#ea4335';
    ctx.font = '16px system-ui, sans-serif';
    ctx.fillText(`${face.label} (${(face.confidence * 100).toFixed(1)}%)`, x, y - 8);

    // 更新结果文本
    resultText.textContent = `${face.label} — ${(face.confidence * 100).toFixed(1)}%`;
    resultText.style.color = isKnown ? '#34a853' : '#ea4335';
}

/**
 * 启动摄像头
 */
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false,
        });
        video.srcObject = stream;
        await video.play();

        isRunning = true;
        btnStart.disabled = true;
        btnStop.disabled = false;
        resultText.textContent = '启动中...';

        // 启动循环发送帧
        function loop() {
            if (!isRunning) return;
            const now = Date.now();
            if (now - lastFrameTime >= FRAME_INTERVAL) {
                lastFrameTime = now;
                sendFrame();
                fpsDisplay.textContent = `FPS: ${(1000 / FRAME_INTERVAL).toFixed(1)}`;
            }
            requestAnimationFrame(loop);
        }
        loop();
    } catch (err) {
        alert('无法访问摄像头，请检查权限设置。\n错误: ' + err.message);
    }
}

/**
 * 停止摄像头
 */
function stopCamera() {
    isRunning = false;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    video.srcObject = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    btnStart.disabled = false;
    btnStop.disabled = true;
    resultText.textContent = '已停止';
    resultText.style.color = '#5f6368';
    fpsDisplay.textContent = 'FPS: 0';
}

// 绑定按钮事件
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);
