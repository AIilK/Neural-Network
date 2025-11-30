import cv2
import os
import json
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
import logging
import speech_recognition as sr
import pyttsx3
import sqlite3
from telegram import Bot
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sounddevice as sd
import subprocess
import speech_recognition as sr

def list_microphones():
    mic_names = sr.Microphone.list_microphone_names()
    for i, mic_name in enumerate(mic_names):
        print(f"Microphone {i}: {mic_name}")

list_microphones()


# ====================== Audio & Environment ======================
os.environ["AUDIODEV"] = "default"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_AUDIODRIVER"] = "pulseaudio"

try:
    print(sd.query_devices())
    sd.default.device = ('USB Audio', 'ALC887-VD Analog')
except Exception as e:
    logging.warning(f"[SoundDevice] {e}")

# ====================== Settings ======================
CAMERA_IDS = [0]
KNOWN_FACES_FOLDER = "people"
UNKNOWN_FOLDER = "unknown_faces"
SIMILARITY_THRESHOLD = 0.65
CLEANUP_HOURS = 24
MAX_TOTAL_UNKNOWN_IMAGES = 100
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DB_PATH = 'events.db'

# ====================== Logging ======================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler("room_monitor.log"),
                        logging.StreamHandler()
                    ])

# ====================== Models ======================
ctx_id = -1
face_model = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=ctx_id)
known_faces = {}

# ====================== TTS ======================
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
if voices:
    female_voice = next((v for v in voices if "female" in v.name.lower()), voices[0])
    tts_engine.setProperty('voice', female_voice.id)
tts_engine.setProperty('rate', 130)
tts_engine.setProperty('volume', 0.9)

def speak_tts(text: str):
    def _run():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"[TTS] {e}")
    threading.Thread(target=_run, daemon=True).start()

# ====================== Telegram ======================
def send_telegram_alert(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logging.info("[Telegram] Message sent successfully")
        except Exception as e:
            logging.error(f"[Telegram] {e}")

# ====================== Database ======================
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

def log_event(event_type, message, cam_id=None, extra=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO events (type, message, time, cam_id, extra) VALUES (?, ?, ?, ?, ?)',
                  (event_type, message, datetime.now().isoformat(), cam_id, json.dumps(extra) if extra else None))
        conn.commit()
        conn.close()
        logging.info(f"[DB] Logged event: {event_type} - {message}")
    except Exception as e:
        logging.error(f"[DB] {e}")

# ====================== Known Faces ======================
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

# ====================== Ollama Chat ======================
SYSTEM_PROMPT = "You are a helpful voice assistant in a meeting room. Respond briefly and politely."

def ollama_chat(user_text: str) -> str:
    try:
        prompt = f"[SYSTEM]\n{SYSTEM_PROMPT}\nUser: {user_text}\nAssistant:"
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        if result.returncode != 0:
            logging.error(f"[Ollama] Error: {result.stderr.decode()}")
            return "Error generating response."
        reply = result.stdout.decode().strip()
        logging.info(f"[Ollama] Reply: {reply}")
        return reply
    except Exception as e:
        logging.error(f"[Ollama] {e}")
        return "I can't respond right now."

# ====================== Speech Recognition ======================
def auto_select_mic_index():
    try:
        names = sr.Microphone.list_microphone_names() or []
        if not names:
            return None
        for i, n in enumerate(names):
            if "mic" in (n or "").lower():
                return i
        return 0
    except Exception as e:
        logging.warning(f"[Mic List] {e}")
        return None

MIC_INDEX = 6

def continuous_listen():
    recognizer = sr.Recognizer()
    logging.info(f"[Mic] Using microphone index: {MIC_INDEX}")
    while True:
        try:
            with sr.Microphone(device_index=MIC_INDEX) as source:
                logging.info(" Listening for voice...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=7, phrase_time_limit=7)
            logging.info(" Processing speech...")
            user_text = recognizer.recognize_google(audio, language='en-US').strip()
            logging.info(f" Heard: {user_text}")
            reply = ollama_chat(user_text)
            logging.info(f"🤖 Ollama: {reply}")
            speak_tts(reply)
            log_event('speech', f'User: {user_text}')
            log_event('chat', f'Ollama: {reply}')
            send_telegram_alert(f'Ollama reply: {reply}')
        except sr.WaitTimeoutError:
            logging.warning("[Mic] Listening timeout")
            continue
        except sr.UnknownValueError:
            logging.warning("[Speech] Could not understand audio")
            continue
        except Exception as e:
            logging.error(f"[Speech] {e}")
            time.sleep(1)

# ====================== Main ======================
def main():
    logging.info("=== Starting Room Monitor ===")
    init_db()
    load_known_faces()
    os.makedirs(UNKNOWN_FOLDER, exist_ok=True)

    logging.info("Testing Ollama connection...")
    print(ollama_chat("Hello, how are you?"))

    threading.Thread(target=continuous_listen, daemon=True).start()
    logging.info("🎧 Always listening and responding...")

    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()

