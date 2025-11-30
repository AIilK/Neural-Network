import cv2
import os
import json
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from playsound import playsound
import threading
import time
import logging

import speech_recognition as sr  # Speech Recognition
import pyttsx3                   # Text-to-speech
import sqlite3
from telegram import Bot
import mediapipe as mp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ====================== Settings ======================
CAMERA_IDS = [0]

KNOWN_FACES_FOLDER = "people"
UNKNOWN_FOLDER = "unknown_faces"
SIMILARITY_THRESHOLD = 0.65
FACE_SAVE_INTERVAL = 5
GREETING_INTERVAL = 3600
CLEANUP_HOURS = 24
USE_GPU = False
WORK_HOURS = (6, 18)
MAX_UNKNOWN_SAVES_PER_PERIOD = 5
UNKNOWN_FACE_PERIOD = 10
MAX_TOTAL_UNKNOWN_IMAGES = 100
ROOM_NAME = "Main Meeting Room"

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------- Gemini Chat ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC8N5VHe_-zV2SUxXASkF5vR2_W6dcCYqs")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

# ---------- Tor / Proxy ----------
USE_TOR_FOR_GEMINI = True  # فقط تماس‌های Gemini از طریق Tor عبور کنند
TOR_SOCKS_PROXY = os.getenv("TOR_SOCKS_PROXY", "socks5h://127.0.0.1:9050")  # اگر Tor Browser است: 127.0.0.1:9150
TOR_GEMINI_TIMEOUT = 45

# دیتابیس
DB_PATH = 'events.db'

# ====================== Logging ======================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler("room_monitor.log"),
                        logging.StreamHandler()
                    ])

# ====================== Models ======================
ctx_id = 0 if USE_GPU else -1
face_model = FaceAnalysis(name="buffalo_l",
                          providers=['CPUExecutionProvider'] if ctx_id == -1 else None)
face_model.prepare(ctx_id=ctx_id)

known_faces = {}
presence_log = {}
unknown_faces_log = []
UNKNOWN_EMBEDDINGS = []
UNKNOWN_FACE_SIMILARITY = 0.7
MAX_SAVES_PER_UNKNOWN_PER_HOUR = 3

# TTS
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
if voices:
    tts_engine.setProperty('voice', voices[0].id)
tts_engine.setProperty('rate', 150)

# Pose
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False,
                             min_detection_confidence=0.5)

chat_active = False
chat_lock = threading.Lock()

# ====================== Helpers ======================
def make_http_session(use_tor=False):
    """
    ساخت یک session با پشتیبانی از پروکسی (Tor)
    """
    s = requests.Session()
    if use_tor:
        s.proxies.update({
            'http': TOR_SOCKS_PROXY,
            'https': TOR_SOCKS_PROXY
        })
    retry = Retry(total=3, backoff_factor=1.2,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def speak_tts(text: str):
    def _run():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"[TTS] {e}")
    threading.Thread(target=_run, daemon=True).start()

def speak_file(file):
    threading.Thread(target=playsound, args=(file,), daemon=True).start()

def in_working_hours():
    hour = datetime.now().hour
    return WORK_HOURS[0] <= hour < WORK_HOURS[1]

def log_event(event_type, message, cam_id=None, extra=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO events (type, message, time, cam_id, extra) VALUES (?, ?, ?, ?, ?)',
                  (event_type, message, datetime.now().isoformat(), cam_id, json.dumps(extra) if extra is not None else None))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"[DB] {e}")

def send_telegram_alert(message):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"[Telegram] {e}")

def cleanup_old_faces():
    now = time.time()
    if not os.path.exists(UNKNOWN_FOLDER):
        return
    for file in os.listdir(UNKNOWN_FOLDER):
        path = os.path.join(UNKNOWN_FOLDER, file)
        if os.path.isfile(path):
            if now - os.path.getctime(path) > CLEANUP_HOURS * 3600:
                try:
                    os.remove(path)
                    logging.info(f"[CLEANUP] Deleted: {file}")
                except Exception as e:
                    logging.error(f"[CLEANUP] {e}")

def save_unknown_face(frame, cam_id):
    os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
    existing_files = [f for f in os.listdir(UNKNOWN_FOLDER) if f.lower().endswith(".jpg")]
    if len(existing_files) >= MAX_TOTAL_UNKNOWN_IMAGES:
        logging.info(f"[INFO] Max unknown images ({MAX_TOTAL_UNKNOWN_IMAGES}) reached.")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(UNKNOWN_FOLDER, f"unknown_cam{cam_id}_{timestamp}.jpg")
    try:
        cv2.imwrite(filename, frame)
        logging.info(f"[INFO] Unknown face saved: {filename}")
        unknown_faces_log.append({"cam_id": cam_id, "time": timestamp, "path": filename})
    except Exception as e:
        logging.error(f"[Unknown Save] {e}")

def load_known_faces():
    logging.info("[INFO] Loading known faces...")
    if not os.path.exists(KNOWN_FACES_FOLDER):
        os.makedirs(KNOWN_FACES_FOLDER)
        return
    for file in os.listdir(KNOWN_FACES_FOLDER):
        name, _ = os.path.splitext(file)
        img_path = os.path.join(KNOWN_FACES_FOLDER, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_model.get(rgb)
        if faces:
            known_faces[name] = faces[0].embedding.reshape(1, -1)
            logging.info(f"[OK] {name} loaded.")

def recognize(embedding):
    embedding = embedding.reshape(1, -1)
    best_name = "Unknown"
    best_score = SIMILARITY_THRESHOLD
    for name, known_emb in known_faces.items():
        sim = cosine_similarity(embedding, known_emb)[0][0]
        if sim > best_score:
            best_score = sim
            best_name = name
    return best_name

def is_same_unknown_face(new_embedding):
    new_embedding = new_embedding.reshape(1, -1)
    now = time.time()
    for unknown in UNKNOWN_EMBEDDINGS:
        sim = cosine_similarity(new_embedding, unknown['embedding'].reshape(1, -1))[0][0]
        if sim > UNKNOWN_FACE_SIMILARITY:
            if now - unknown['last_seen'] > 3600:
                unknown['save_count'] = 0
            unknown['last_seen'] = now
            return unknown
    return None

# ====================== Gemini Chat ======================
SYSTEM_PROMPT = (
    "You are a helpful voice assistant in a meeting room. "
    "Respond briefly, politely, and helpfully. "
    "If a user asks a task-like question, give step-by-step guidance."
)

def gemini_chat(user_text: str, system_prompt: str = SYSTEM_PROMPT, max_tokens: int = 300, temperature: float = 0.6) -> str:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "REPLACE_WITH_YOUR_KEY":
        return "Gemini API key is not configured."

    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    contents = []
    if system_prompt:
        contents.append({"role": "user", "parts": [{"text": f"[SYSTEM]\n{system_prompt}"}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    session = make_http_session(USE_TOR_FOR_GEMINI)
    timeout = TOR_GEMINI_TIMEOUT if USE_TOR_FOR_GEMINI else 20

    try:
        r = session.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=timeout)
        if r.status_code != 200:
            logging.warning(f"[Gemini] {r.status_code} {r.text[:200]}")
            return f"Gemini API error {r.status_code}"
        data = r.json()
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            reply = "\n".join([p.get("text", "") for p in parts]).strip()
            return reply or "I didn't get a response."
        return "I didn't get a response."
    except Exception as e:
        logging.error(f"[Gemini] {e}")
        return "I can't reach Gemini right now."

# ====================== Speech Recognition ======================
def auto_select_mic_index(prefer_keywords=("webcam", "camera", "usb")):
    try:
        names = sr.Microphone.list_microphone_names() or []
        if not names:
            return None
        for kw in prefer_keywords:
            for i, n in enumerate(names):
                if kw.lower() in (n or "").lower():
                    return i
        return None
    except Exception as e:
        logging.warning(f"[Mic List] {e}")
        return None

MIC_INDEX = auto_select_mic_index()

def recognize_voice(cam_id=None):
    recognizer = sr.Recognizer()
    try:
        try:
            names = sr.Microphone.list_microphone_names() or []
            print("\n=== Microphone list ===")
            for i, n in enumerate(names):
                mark = " (auto-selected)" if (MIC_INDEX is not None and i == MIC_INDEX) else ""
                print(f"{i}: {n}{mark}")
            print(f"Using MIC_INDEX={MIC_INDEX} (None means system default)\n")
        except Exception:
            pass

        with sr.Microphone(device_index=MIC_INDEX) as source:
            print("🎤 Listening... speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=7)

        print("⌛ Recognizing (English)...")
        user_text = recognizer.recognize_google(audio, language='en-US').strip()
        print(f"✅ You said: {user_text}")

        reply = gemini_chat(user_text)
        print(f"🤖 Gemini: {reply}")
        speak_tts(reply)

        if cam_id is not None:
            log_event('speech', f'User: {user_text}', cam_id)
            log_event('chat', f'Gemini: {reply}', cam_id)
            send_telegram_alert(f'Gemini reply on cam {cam_id}: {reply}')

    except sr.WaitTimeoutError:
        print("⏳ No speech detected (timeout).")
    except sr.UnknownValueError:
        print("❌ Could not understand the audio.")
        speak_tts("Sorry, I didn't catch that.")
    except sr.RequestError as e:
        print(f"❌ Speech service error: {e}")
        speak_tts("Speech recognition is not available.")
    except Exception as e:
        logging.info(f"[Speech] Error: {e}")

# ====================== Camera & Events ======================
def process_camera(cam_id):
    global chat_active
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        logging.error(f"[Cam {cam_id}] cannot open.")
        return

    last_announced = {}
    prev_gray = None
    motion_detected = False
    motion_counter = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_model.get(rgb)
        seen_names = []
        now_ts = time.time()

        if frame_count % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                non_zero = np.count_nonzero(diff > 30)
                if non_zero > 5000:
                    if not motion_detected:
                        log_event('motion', 'Motion detected', cam_id)
                        send_telegram_alert(f'Motion detected on camera {cam_id}')
                        motion_detected = True
                        motion_counter = 0
                else:
                    motion_counter += 1
                    if motion_counter > 10:
                        motion_detected = False
            prev_gray = gray

        if frame_count % 10 == 0:
            results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                rw = results.pose_landmarks.landmark[16]
                rs = results.pose_landmarks.landmark[12]
                if rw.y < rs.y:
                    with chat_lock:
                        if not chat_active:
                            chat_active = True
                            logging.info("[CHAT] Activated by gesture")
                            threading.Thread(target=lambda: (recognize_voice(cam_id), set_chat_inactive()), daemon=True).start()
                    log_event('pose', 'Hand raised detected', cam_id)
                    send_telegram_alert(f'Hand raised on camera {cam_id}')

        if frame_count % 150 == 0:
            threading.Thread(target=recognize_voice, args=(cam_id,), daemon=True).start()

        for face in faces:
            name = recognize(face.embedding)
            seen_names.append(name)
            if name == "Unknown":
                unk = is_same_unknown_face(face.embedding)
                if unk:
                    if unk.get('save_count', 0) < MAX_SAVES_PER_UNKNOWN_PER_HOUR:
                        save_unknown_face(frame, cam_id)
                        unk['save_count'] = unk.get('save_count', 0) + 1
                else:
                    UNKNOWN_EMBEDDINGS.append({
                        'embedding': face.embedding,
                        'last_seen': now_ts,
                        'save_count': 1
                    })
                    save_unknown_face(frame, cam_id)

                if in_working_hours():
                    speak_file("welcome_unknown.mp3")
                else:
                    speak_file("alert.mp3")
            else:
                if name not in last_announced or now_ts - last_announced[name] > GREETING_INTERVAL:
                    speak_tts(f"Hello {name}")
                    last_announced[name] = now_ts
                presence_log[name] = {"last_seen": now_ts, "present": True, "cam_id": cam_id}

        for tracked_name in list(presence_log.keys()):
            if presence_log[tracked_name]["cam_id"] == cam_id and tracked_name not in seen_names:
                if time.time() - presence_log[tracked_name]["last_seen"] > 10:
                    presence_log[tracked_name]["present"] = False

        time.sleep(0.1)

    cap.release()

def set_chat_inactive():
    global chat_active
    chat_active = False

def report_room_status():
    print("\n======= Presence Report for {} =======".format(ROOM_NAME))
    present = [n for n, i in presence_log.items() if i["present"]]
    absent = [n for n, i in presence_log.items() if not i["present"]]
    if present:
        for n in present:
            cam_id = presence_log[n]["cam_id"]
            last_seen = datetime.fromtimestamp(presence_log[n]["last_seen"]).strftime("%H:%M:%S")
            print(f"✅ {n} (cam {cam_id}, last seen: {last_seen})")
    else:
        print("✅ No one present")
    if absent:
        print(f"🚪 Absent: {', '.join(absent)}")
    else:
        print("🚪 No one left")
    print("================================\n")

    if unknown_faces_log:
        print("⚠️ Unknown faces:")
        for entry in unknown_faces_log[-10:]:
            print(f"  cam {entry['cam_id']}, time: {entry['time']}, file: {entry['path']}")
    print("================================\n")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT,
        message TEXT,
        time TEXT,
        cam_id INTEGER,
        extra TEXT
    )''')
    conn.commit()
    conn.close()

def cleanup_unknown_faces_daily():
    now = time.time()
    global UNKNOWN_EMBEDDINGS
    UNKNOWN_EMBEDDINGS = [u for u in UNKNOWN_EMBEDDINGS if now - u['last_seen'] < 24*3600]

# ====================== Main ======================
def main():
    load_known_faces()
    os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
    init_db()

    # شروع نخ‌های دوربین
    for cam_id in CAMERA_IDS:
        threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()

    last_cleanup_time = 0
    while True:
        try:
            now = time.time()
            cleanup_old_faces()
            report_room_status()

            if now - last_cleanup_time > 24 * 3600:
                cleanup_unknown_faces_daily()
                last_cleanup_time = now

            time.sleep(300)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"[Main Loop] Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
