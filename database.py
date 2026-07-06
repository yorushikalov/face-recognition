"""
database.py — 用户数据库模块（SQLite）

功能：用户注册、登录验证、人脸特征存取、短信验证码
"""
import sqlite3
import pickle
import random
import string
import numpy as np
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "data/users.db"


# ---------- 初始化 ----------
def init_db():
    """创建数据库表"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        phone TEXT UNIQUE NOT NULL,
        real_name TEXT NOT NULL,
        face_embedding BLOB,
        face_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS sms_codes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone TEXT NOT NULL,
        code TEXT NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        used INTEGER DEFAULT 0
    )""")
    conn.commit()
    conn.close()


# ---------- 用户 CRUD ----------
def create_user(username, password, phone, real_name, face_embedding=None, face_count=0):
    """创建新用户，返回 (success, message)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        emb_blob = pickle.dumps(face_embedding) if face_embedding is not None else None
        c.execute(
            "INSERT INTO users (username, password_hash, phone, real_name, face_embedding, face_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, password_hash, phone, real_name, emb_blob, face_count),
        )
        conn.commit()
        return True, "注册成功"
    except sqlite3.IntegrityError as e:
        msg = str(e)
        if "username" in msg:
            return False, "用户名已存在"
        elif "phone" in msg:
            return False, "手机号已被注册"
        return False, "注册失败"
    finally:
        conn.close()


def verify_password(username, password):
    """验证用户名密码，返回 (success, user_dict_or_message)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()

    if row is None:
        return False, "用户名不存在"

    if check_password_hash(row[2], password):
        return True, _row_to_dict(row)
    return False, "密码错误"


def verify_phone(phone):
    """验证手机号是否已注册，返回 (success, user_dict_or_message)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE phone = ?", (phone,))
    row = c.fetchone()
    conn.close()

    if row is None:
        return False, "手机号未注册"
    return True, _row_to_dict(row)


def get_user_by_id(user_id):
    """根据 ID 获取用户"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_all_face_users():
    """获取所有已录入人脸的用户（用于人脸识别登录）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE face_embedding IS NOT NULL")
    rows = c.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def update_face_embedding(user_id, embedding, face_count):
    """更新用户的人脸特征向量"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_blob = pickle.dumps(embedding)
    c.execute(
        "UPDATE users SET face_embedding = ?, face_count = ? WHERE id = ?",
        (emb_blob, face_count, user_id),
    )
    conn.commit()
    conn.close()


# ---------- 短信验证码 ----------
def generate_sms_code(phone):
    """生成 6 位验证码，返回验证码字符串"""
    code = "".join(random.choices(string.digits, k=6))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    expires_at = datetime.now() + timedelta(minutes=5)
    c.execute(
        "INSERT INTO sms_codes (phone, code, expires_at) VALUES (?, ?, ?)",
        (phone, code, expires_at),
    )
    conn.commit()
    conn.close()
    return code


def verify_sms_code(phone, code):
    """验证短信验证码"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT * FROM sms_codes WHERE phone = ? AND code = ? AND used = 0 "
        "ORDER BY id DESC LIMIT 1",
        (phone, code),
    )
    row = c.fetchone()
    if row is None:
        conn.close()
        return False, "验证码错误"

    expires_at = datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S.%f")
    if datetime.now() > expires_at:
        c.execute("UPDATE sms_codes SET used = 1 WHERE id = ?", (row[0],))
        conn.commit()
        conn.close()
        return False, "验证码已过期"

    c.execute("UPDATE sms_codes SET used = 1 WHERE id = ?", (row[0],))
    conn.commit()
    conn.close()
    return True, "验证成功"


# ---------- 辅助函数 ----------
def _row_to_dict(row):
    return {
        "id": row[0],
        "username": row[1],
        "password_hash": row[2],
        "phone": row[3],
        "real_name": row[4],
        "face_embedding": pickle.loads(row[5]) if row[5] else None,
        "face_count": row[6],
        "created_at": row[7],
    }
