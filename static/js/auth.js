/**
 * auth.js — 登录/注册前端逻辑
 * 支持：密码登录、短信登录、人脸识别登录、账号注册+人脸采集
 */

// ===== 通用工具 =====
function showMessage(msg, type) {
    const el = document.getElementById("auth-message");
    el.textContent = msg;
    el.className = "auth-message " + type;
    el.style.display = "block";
}
function hideMessage() {
    document.getElementById("auth-message").style.display = "none";
}

// ===== 登录方式切换 =====
document.addEventListener("DOMContentLoaded", () => {
    const switchBtns = document.querySelectorAll(".switch-btn");
    if (!switchBtns.length) return;

    switchBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            // 更新按钮状态
            switchBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            // 隐藏所有表单
            document.querySelectorAll(".auth-form").forEach((f) => (f.style.display = "none"));

            // 显示目标表单
            const targetId = btn.dataset.target;
            document.getElementById(targetId).style.display = "block";

            // 如果切换到人脸识别，启动摄像头
            if (targetId === "form-face") initCamera();
            else stopCamera();

            hideMessage();
        });
    });
});

// ===== 账号密码登录 =====
async function loginPassword() {
    hideMessage();
    const username = document.getElementById("login-username").value.trim();
    const password = document.getElementById("login-password").value;

    if (!username || !password) {
        showMessage("请输入用户名和密码", "error");
        return;
    }

    try {
        const res = await fetch("/api/login_password", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (data.success) {
            showMessage("登录成功，正在跳转...", "success");
            setTimeout(() => (window.location.href = "/dashboard"), 800);
        } else {
            showMessage(data.message, "error");
        }
    } catch (e) {
        showMessage("网络错误，请重试", "error");
    }
}

// 回车登录
document.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        const formPassword = document.getElementById("form-password");
        if (formPassword && formPassword.style.display !== "none") {
            loginPassword();
        }
    }
});

// ===== 短信验证码登录 =====
let smsCountdown = 0;

async function sendSMS() {
    const phone = document.getElementById("sms-phone").value.trim();
    if (!phone || phone.length !== 11) {
        showMessage("请输入正确的 11 位手机号", "error");
        return;
    }

    try {
        const res = await fetch("/api/send_sms", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ phone }),
        });
        const data = await res.json();
        if (data.success) {
            showMessage(data.message, "success");
            // 倒计时
            smsCountdown = 60;
            const btn = document.getElementById("btn-send-sms");
            btn.disabled = true;
            const timer = setInterval(() => {
                smsCountdown--;
                btn.textContent = smsCountdown + "s 后重发";
                if (smsCountdown <= 0) {
                    clearInterval(timer);
                    btn.disabled = false;
                    btn.textContent = "获取验证码";
                }
            }, 1000);
        } else {
            showMessage(data.message, "error");
        }
    } catch (e) {
        showMessage("网络错误，请重试", "error");
    }
}

async function loginSMS() {
    hideMessage();
    const phone = document.getElementById("sms-phone").value.trim();
    const code = document.getElementById("sms-code").value.trim();

    if (!phone || !code) {
        showMessage("请输入手机号和验证码", "error");
        return;
    }

    try {
        const res = await fetch("/api/login_sms", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ phone, code }),
        });
        const data = await res.json();
        if (data.success) {
            showMessage("登录成功，正在跳转...", "success");
            setTimeout(() => (window.location.href = "/dashboard"), 800);
        } else {
            showMessage(data.message, "error");
        }
    } catch (e) {
        showMessage("网络错误，请重试", "error");
    }
}

// ===== 摄像头管理 =====
let videoStream = null;
let faceInterval = null;

async function initCamera() {
    const video = document.getElementById("face-video");
    if (!video) return;

    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
        });
        video.srcObject = videoStream;
    } catch (e) {
        showMessage("无法访问摄像头，请检查权限设置", "error");
    }
}

function stopCamera() {
    if (faceInterval) { clearInterval(faceInterval); faceInterval = null; }
    if (videoStream) {
        videoStream.getTracks().forEach((t) => t.stop());
        videoStream = null;
    }
}

function captureFrame() {
    const video = document.getElementById("face-video");
    const canvas = document.getElementById("face-canvas");
    if (!video || !canvas || video.readyState < 2) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    return canvas.toBlob
        ? awaitBlob(canvas)
        : null;
}

function awaitBlob(canvas) {
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
}

// ===== 人脸识别登录 =====
async function startFaceLogin() {
    hideMessage();
    const status = document.getElementById("face-status");
    const btn = document.getElementById("btn-face-login");

    if (!videoStream) await initCamera();
    status.textContent = "正在扫描人脸...";
    status.className = "face-status scanning";
    btn.disabled = true;
    btn.textContent = "识别中...";

    // 延迟一下让摄像头稳定
    await new Promise((r) => setTimeout(r, 500));

    const canvas = document.getElementById("face-canvas");
    const video = document.getElementById("face-video");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    const blob = await awaitBlob(canvas);
    if (!blob) {
        status.textContent = "图片采集失败";
        status.className = "face-status error";
        btn.disabled = false;
        btn.textContent = "开始人脸识别";
        return;
    }

    const formData = new FormData();
    formData.append("image", blob, "face.jpg");

    try {
        const res = await fetch("/api/login_face", { method: "POST", body: formData });
        const data = await res.json();

        if (data.success) {
            status.textContent = data.message;
            status.className = "face-status success";
            setTimeout(() => (window.location.href = "/dashboard"), 1000);
        } else {
            status.textContent = data.message;
            status.className = "face-status error";
            btn.disabled = false;
            btn.textContent = "重新识别";
        }
    } catch (e) {
        status.textContent = "网络错误，请重试";
        status.className = "face-status error";
        btn.disabled = false;
        btn.textContent = "开始人脸识别";
    }
}

// ===== 注册 =====
async function doRegister() {
    hideMessage();
    const username = document.getElementById("reg-username").value.trim();
    const password = document.getElementById("reg-password").value;
    const real_name = document.getElementById("reg-name").value.trim();
    const phone = document.getElementById("reg-phone").value.trim();

    if (!username || !password || !real_name || !phone) {
        showMessage("请填写所有字段", "error");
        return;
    }

    try {
        const res = await fetch("/api/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password, phone, real_name }),
        });
        const data = await res.json();
        if (data.success) {
            showMessage(data.message, "success");
            // 显示人脸采集步骤
            document.getElementById("step-form").style.display = "none";
            document.getElementById("step-face").style.display = "block";
            document.getElementById("auth-message").style.display = "block";
        } else {
            showMessage(data.message, "error");
        }
    } catch (e) {
        showMessage("网络错误，请重试", "error");
    }
}

// ===== 人脸采集 =====
let faceCaptureCount = 0;

async function startFaceCapture() {
    hideMessage();
    await initCamera();

    const btnCapture = document.getElementById("btn-capture-face");
    const btnStop = document.getElementById("btn-stop-face");
    const btnFinish = document.getElementById("btn-finish-reg");
    const progressBar = document.getElementById("face-progress");
    const status = document.getElementById("face-status");

    btnCapture.style.display = "none";
    btnStop.style.display = "block";
    btnFinish.style.display = "none";
    progressBar.style.display = "block";
    status.textContent = "正在自动采集人脸...";
    status.className = "face-status scanning";

    // 每 0.8 秒自动采集一张
    faceInterval = setInterval(async () => {
        const canvas = document.getElementById("face-canvas");
        const video = document.getElementById("face-video");

        if (video.readyState < 2) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);

        const blob = await awaitBlob(canvas);
        if (!blob) return;

        const formData = new FormData();
        formData.append("image", blob, "face.jpg");

        try {
            const res = await fetch("/api/register_face", { method: "POST", body: formData });
            const data = await res.json();
            if (data.success) {
                faceCaptureCount = data.face_count;
                const pct = Math.min(faceCaptureCount * 10, 100);
                document.getElementById("progress-fill").style.width = pct + "%";
                document.getElementById("progress-text").textContent =
                    "已采集 " + faceCaptureCount + " 张（建议至少 10 张）";

                if (faceCaptureCount >= 15) {
                    // 自动停止
                    stopFaceCapture();
                    btnFinish.style.display = "block";
                    status.textContent = "采集完成！点击下方按钮保存";
                    status.className = "face-status success";
                }
            }
        } catch (e) {
            console.error("Face capture error:", e);
        }
    }, 800);
}

function stopFaceCapture() {
    if (faceInterval) { clearInterval(faceInterval); faceInterval = null; }
    const btnCapture = document.getElementById("btn-capture-face");
    const btnStop = document.getElementById("btn-stop-face");
    const btnFinish = document.getElementById("btn-finish-reg");

    btnCapture.style.display = "block";
    btnCapture.textContent = "继续采集";
    btnStop.style.display = "none";

    if (faceCaptureCount >= 3) {
        btnFinish.style.display = "block";
    }
}

async function finishRegistration() {
    stopCamera();
    try {
        const res = await fetch("/api/register_finish", { method: "POST" });
        const data = await res.json();
        if (data.success) {
            showMessage("注册完成！即将跳转登录页...", "success");
            document.getElementById("step-face").style.display = "none";
            setTimeout(() => (window.location.href = "/login"), 1500);
        } else {
            showMessage(data.message, "error");
        }
    } catch (e) {
        showMessage("网络错误，请重试", "error");
    }
}

// 页面离开时关闭摄像头
window.addEventListener("beforeunload", stopCamera);
