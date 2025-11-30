# -*- coding: utf-8 -*-
import os, cv2, time, json, math, queue, threading, logging, sqlite3, requests, numpy as np
from datetime import datetime
from collections import deque, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# صوت/گفتار
import speech_recognition as sr
import pyttsx3

# بینایی
from insightface.app import FaceAnalysis
import mediapipe as mp

# تلگرام
from telegram import Bot
from dotenv import load_dotenv

# ------------- تنظیمات عمومی ------------------
# متغیرهای محیطی (کلیدها را در env ست کنید؛ در کد ننویسید)
def require_env(name: str, optional=False, default=None): 
    val = os.getenv(name, default)
    if not val and not optional:
        raise RuntimeError(f"Missing environment variable: {name}")
    
    return val

# Provider گفت‌وگو: OPENAI یا GEMINI
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "OPENAI").upper().strip()

# OpenAI
load_dotenv()
OPENAI_API_KEY = require_env("OPENAI_API_KEY", optional=(CHAT_PROVIDER!="OPENAI"))
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Gemini (Google)
GEMINI_API_KEY = require_env("GEMINI_API_KEY", optional=(CHAT_PROVIDER!="GEMINI"))
# REST endpoint رسمی Gemini (v1beta)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# تلگرام
TELEGRAM_TOKEN  = require_env("8307454840:AAEwojyScvS1O5gul-XFvKx46bh005eU5j4", optional=True)
TELEGRAM_CHAT_ID= require_env("6590467605", optional=True)

# مسیرها
KNOWN_FACES_FOLDER = "people"
UNKNOWN_FOLDER     = "unknown_faces"
DB_PATH            = "events.db"
LOG_PATH           = "room_monitor.log"

# دوربین‌ها و میکروفون
CAMERA_IDS = [0, 1, 2]             # آیدی‌های دوربین
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "-1"))  # -1 یعنی پیش‌فرض سیستم

# آستانه‌ها
SIMILARITY_THRESHOLD = 0.68        # آستانه تشخیص فرد شناخته‌شده (پس از نرمال‌سازی)
UNKNOWN_FACE_SIMILARITY = 0.72     # برای تشخیص تکراری بودن ناشناس
MAX_SAVES_PER_UNKNOWN_PER_HOUR = 3
MAX_TOTAL_UNKNOWN_IMAGES = 100
GREETING_INTERVAL = 3600           # ثانیه
WORK_HOURS = (6, 18)               # ساعت کاری
CLEANUP_HOURS = 24                 # حذف تصاویر ناشناس قدیمی
UNKNOWN_FACE_PERIOD = 10           # برای دریافت/ذخیره‌ی ناشناس‌ها (ثانیه)
ROOM_NAME = "اتاق جلسات مرکزی"

# کارایی و تحلیل
USE_GPU = False                    # این نسخه CPU-friendly است
DET_SIZE = (640, 640)              # سایز ورودی آشکارساز چهره
DET_THRESH = 0.5                   # آستانه تشخیص چهره
FRAME_SKIP_FACE = 1                # هر چند فریم یک‌بار چهره
FRAME_SKIP_POSE = 10               # هر چند فریم یک‌بار pose
MOTION_DIFF_PIXELS = 0.02          # درصد پیکسل‌های متفاوت برای حرکت (نسبت به کل)

# لاگینگ
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH),
                              logging.StreamHandler()])

# ------------- وضعیت و حافظه‌های سراسری (با قفل) ------------------
presence_log = {}                     # name -> {last_seen, present, cam_id}
presence_lock = threading.Lock()

unknown_faces_log = deque(maxlen=1000)
unknown_embeddings = []               # [{'embedding': np.array, 'last_seen': ts, 'save_count': int}]
unknown_lock = threading.Lock()

last_greet = defaultdict(lambda: 0.0) # name -> ts
greet_lock = threading.Lock()

# شناخته‌شده‌ها (برداری شده)
# نام → لیست embedding های نرمال
known_dict = defaultdict(list)
# ماتریس همه‌ی embedding ها و لیبل‌ها (برای سرعت)
known_matrix = None  # shape: (N, D)
known_labels = []    # length N
known_lock = threading.Lock()

# صف‌ها و workerها
event_queue = queue.Queue()
notify_queue = queue.Queue()
tts_queue = queue.Queue()

shutdown_event = threading.Event()

# TTS
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
if voices:
    tts_engine.setProperty('voice', voices[0].id)
tts_engine.setProperty('rate', 150)

# Pose
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# ------------- ابزارهای کمکی ------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def in_working_hours():
    hour = datetime.now().hour
    return WORK_HOURS[0] <= hour < WORK_HOURS[1]

def rebuild_index():
    global known_matrix, known_labels
    with known_lock:
        all_embs = []
        all_labels = []
        for name, embs in known_dict.items():
            for e in embs:
                all_embs.append(e)
                all_labels.append(name)
        if all_embs:
            known_matrix = np.vstack(all_embs)  # (N, D)
            known_labels = all_labels
        else:
            known_matrix = None
            known_labels = []

def recognize_face(embedding: np.ndarray):
    """
    embedding: (D,) raw (we'll normalize)
    returns: (best_name, best_score)
    """
    e = l2_normalize(embedding).reshape(1, -1)
    with known_lock:
        if known_matrix is None or known_matrix.shape[0] == 0:
            return ("Unknown", 0.0)
        # cosine similarity: (1xD)·(DxN) = (1xN)
        sims = (e @ known_matrix.T).ravel()
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        name = known_labels[idx]
        if score >= SIMILARITY_THRESHOLD:
            return (name, score)
        return ("Unknown", score)

def is_same_unknown_face(new_embedding: np.ndarray):
    e = l2_normalize(new_embedding).reshape(1, -1)
    now = time.time()
    with unknown_lock:
        for u in unknown_embeddings:
            sim = float((e @ u['embedding'].reshape(1, -1).T).ravel()[0])
            if sim > UNKNOWN_FACE_SIMILARITY:
                if now - u['last_seen'] > 3600:
                    u['save_count'] = 0
                u['last_seen'] = now
                return u
        # اگر نبود:
        unknown_embeddings.append({
            'embedding': l2_normalize(new_embedding),
            'last_seen': now,
            'save_count': 0
        })
        return None

def cleanup_unknown_faces_daily():
    now = time.time()
    with unknown_lock:
        unknown_embeddings[:] = [u for u in unknown_embeddings if now - u['last_seen'] < 24*3600]

def save_unknown_face(frame, cam_id):
    os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
    existing_files = [f for f in os.listdir(UNKNOWN_FOLDER) if f.endswith(".jpg")]
    if len(existing_files) >= MAX_TOTAL_UNKNOWN_IMAGES:
        logging.info(f"[INFO] حداکثر تعداد تصاویر ناشناس ({MAX_TOTAL_UNKNOWN_IMAGES}) رسیده است.")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(UNKNOWN_FOLDER, f"unknown_cam{cam_id}_{timestamp}.jpg")
    try:
        cv2.imwrite(filename, frame)
        unknown_faces_log.append({"cam_id": cam_id, "time": timestamp, "path": filename})
        logging.info(f"[INFO] تصویر ناشناس ذخیره شد: {filename}")
    except Exception as e:
        logging.error(f"[Unknown Save] {e}")

def list_microphones():
    # کمک برای یافتن index میکروفون دوربین (یک‌بار دستی اجرا کنید و MIC_DEVICE_INDEX را ست کنید)
    try:
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{i}: {name}")
    except Exception as e:
        logging.warning(f"[Mic List] {e}")

def log_event(event_type, message, cam_id=None, extra=None):
    event_queue.put({
        "type": event_type,
        "message": message,
        "time": datetime.now().isoformat(),
        "cam_id": cam_id,
        "extra": json.dumps(extra) if extra is not None else None
    })

def send_telegram_alert_async(text):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        notify_queue.put({"type": "telegram", "text": text})

def speak_tts_async(text):
    tts_queue.put(text)

# ------------- Workerها ------------------
def db_worker():
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT,
        message TEXT,
        time TEXT,
        cam_id INTEGER,
        extra TEXT
    )''')
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    conn.commit()
    while not shutdown_event.is_set():
        try:
            evt = event_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            c.execute('INSERT INTO events (type, message, time, cam_id, extra) VALUES (?, ?, ?, ?, ?)',
                      (evt['type'], evt['message'], evt['time'], evt['cam_id'], evt['extra']))
            conn.commit()
        except Exception as e:
            logging.error(f"[DB] {e}")
    conn.close()

def notify_worker():
    bot = None
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
        except Exception as e:
            logging.error(f"[Telegram Init] {e}")
    while not shutdown_event.is_set():
        try:
            job = notify_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if job["type"] == "telegram" and bot:
            try:
                bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=job["text"])
            except Exception as e:
                logging.error(f"[Telegram] {e}")

def tts_worker():
    while not shutdown_event.is_set():
        try:
            text = tts_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"[TTS] {e}")

def audio_worker():
    recognizer = sr.Recognizer()
    # کالیبراسیون آستانه انرژی برای محیط
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6
    recognizer.non_speaking_duration = 0.3
    # انتخاب میکروفون دوربین
    mic_kwargs = {}
    if MIC_DEVICE_INDEX >= 0:
        mic_kwargs["device_index"] = MIC_DEVICE_INDEX
    try:
        with sr.Microphone(**mic_kwargs) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
    except Exception as e:
        logging.error(f"[Mic] دسترسی به میکروفون: {e}")
        return

    while not shutdown_event.is_set():
        try:
            with sr.Microphone(**mic_kwargs) as source:
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=4)
            text = ""
            try:
                # vosk اگر نصب و مدل دارید
                text = recognizer.recognize_vosk(audio, language='fa')
            except Exception:
                try:
                    text = recognizer.recognize_vosk(audio, language='en')
                except Exception:
                    # fallback (نیازمند اینترنت)
                    text = recognizer.recognize_google(audio, language='fa-IR')
            text = (text or "").strip()
            if text:
                log_event('speech', f'گفتار: {text}')
                send_telegram_alert_async(f'گفتار شناسایی شد: {text}')
                # اگر گفت‌وگو فعال باشد، پاسخ تولید شود
                reply = chat_respond(text)
                if reply:
                    speak_tts_async(reply)
        except Exception as e:
            logging.debug(f"[Speech] {e}")
        time.sleep(0.2)

# ------------- گفت‌وگو (Chat) ------------------
def openai_chat(message, system_prompt=None, max_tokens=300, temperature=0.6):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": ([{"role":"system","content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": message}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    for attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=payload, timeout=15)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                logging.warning(f"[OpenAI] {r.status_code} {r.text[:200]}")
        except Exception as e:
            logging.warning(f"[OpenAI Attempt {attempt+1}] {e}")
        time.sleep(1.5)
    return "در حال حاضر قادر به دریافت پاسخ نیستم."

def gemini_chat(message, system_prompt=None, max_tokens=300, temperature=0.6):
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    # فرمت generateContent
    contents = []
    if system_prompt:
        contents.append({"role":"user", "parts":[{"text": f"[SYSTEM]\n{system_prompt}"}]})
    contents.append({"role":"user", "parts":[{"text": message}]})
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    for attempt in range(3):
        try:
            r = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=15)
            if r.status_code == 200:
                data = r.json()
                # مسیر متن خروجی
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    txt = "\n".join([p.get("text","") for p in parts]).strip()
                    return txt or "پاسخی دریافت نشد."
            else:
                logging.warning(f"[Gemini] {r.status_code} {r.text[:200]}")
        except Exception as e:
            logging.warning(f"[Gemini Attempt {attempt+1}] {e}")
        time.sleep(1.5)
    return "در حال حاضر قادر به دریافت پاسخ نیستم."

def chat_respond(user_text: str) -> str:
    system_prompt = (
        "تو دستیار صوتی اتاق جلسات هستی؛ پاسخ‌ها کوتاه، محترمانه و کاربردی باشند. "
        "اگر پرسش عمومی بود، خلاصه‌وار پاسخ بده؛ اگر کاری خواست، راهنمایی مرحله‌به‌مرحله بده."
    )
    if CHAT_PROVIDER == "OPENAI" and OPENAI_API_KEY:
        return openai_chat(user_text, system_prompt=system_prompt)
    elif CHAT_PROVIDER == "GEMINI" and GEMINI_API_KEY:
        return gemini_chat(user_text, system_prompt=system_prompt)
    else:
        return "پیکربندی سامانهٔ گفت‌وگو کامل نیست."

# ------------- بینایی: InsightFace & Pose ------------------
def init_face_model():
    ctx_id = 0 if USE_GPU else -1
    # buffalo_sc برای CPU مناسب و دقیق
    app = FaceAnalysis(name="buffalo_sc",
                       providers=['CPUExecutionProvider'] if ctx_id == -1 else None)
    app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
    return app

face_app = init_face_model()

def load_known_faces():
    """
    ساختار پوشه‌ها:
    people/
      ├─ ali/ img1.jpg, img2.png, ...
      ├─ sara/ ...
      └─ mohammad.jpg   (حالت تک‌فایله)
    """
    logging.info("[INFO] بارگذاری چهره‌های شناخته‌شده...")
    if not os.path.exists(KNOWN_FACES_FOLDER):
        os.makedirs(KNOWN_FACES_FOLDER)
        return
    count = 0
    for entry in os.listdir(KNOWN_FACES_FOLDER):
        p = os.path.join(KNOWN_FACES_FOLDER, entry)
        if os.path.isdir(p):
            name = entry
            for f in os.listdir(p):
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                    img = cv2.imread(os.path.join(p,f))
                    if img is None: continue
                    faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if faces:
                        emb = l2_normalize(faces[0].embedding)
                        known_dict[name].append(emb)
                        count += 1
        else:
            # حالت تک‌فایله
            if entry.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                name,_ = os.path.splitext(entry)
                img = cv2.imread(p)
                if img is None: continue
                faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if faces:
                    emb = l2_normalize(faces[0].embedding)
                    known_dict[name].append(emb)
                    count += 1
    # میانگین‌گیری اختیاری (اگر از هر نفر نمونه زیاد دارید، می‌توانید نگه دارید؛
    # ما همان لیست را نگه می‌داریم تا پوشش تغییرات بیشتر باشد)
    rebuild_index()
    logging.info(f"[INFO] {count} نمونه چهره بارگذاری شد. افراد: {len(known_dict)}")

# ------------- پردازش دوربین ------------------
def process_camera(cam_id):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        logging.error(f"[Cam {cam_id}] باز نشد.")
        return
    prev_gray = None
    motion_detected = False
    motion_cool = 0
    frame_idx = 0
    last_unknown_ts = 0.0

    while not shutdown_event.is_set():
        ok, frame = cap.read()
        if not ok:
            logging.warning(f"[Cam {cam_id}] read fail—retrying...")
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(cam_id)
            time.sleep(0.5)
            continue

        h, w = frame.shape[:2]
        frame_idx += 1

        # --- Motion Detection (هر 5 فریم) ---
        if frame_idx % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                nz = np.count_nonzero(diff > 25)
                if nz > MOTION_DIFF_PIXELS * gray.size:
                    if not motion_detected:
                        log_event('motion', 'حرکت شناسایی شد', cam_id)
                        send_telegram_alert_async(f'حرکت در دوربین {cam_id}')
                    motion_detected = True
                    motion_cool = 0
                else:
                    motion_cool += 1
                    if motion_cool > 10:
                        motion_detected = False
            prev_gray = gray

        # --- Pose Detection (هر FRAME_SKIP_POSE فریم) ---
        if frame_idx % FRAME_SKIP_POSE == 0:
            results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                rw = results.pose_landmarks.landmark[16]
                rs = results.pose_landmarks.landmark[12]
                if rw.y < rs.y:
                    log_event('pose', 'دست راست بالا شناسایی شد', cam_id)
                    send_telegram_alert_async(f'ژست دست در دوربین {cam_id}')
                    # پاسخ گفت‌وگویی سبک (Trigger)
                    reply = chat_respond("سلام")
                    if reply:
                        speak_tts_async(reply)

        # --- Face Detection/Recognition ---
        if frame_idx % FRAME_SKIP_FACE == 0:
            faces = face_app.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            seen_names = []
            now = time.time()

            for f in faces:
                name, score = recognize_face(f.embedding)
                if name == "Unknown":
                    # کنترل ذخیره‌ی زیاد ناشناس‌ها
                    same = is_same_unknown_face(f.embedding)
                    # فاصله زمانی عمومی بین ذخیره‌ها
                    if now - last_unknown_ts >= UNKNOWN_FACE_PERIOD:
                        should_save = False
                        if same:
                            if same['save_count'] < MAX_SAVES_PER_UNKNOWN_PER_HOUR:
                                should_save = True
                                same['save_count'] += 1
                        else:
                            should_save = True
                            # مورد جدید (اولین ذخیره)
                            with unknown_lock:
                                unknown_embeddings[-1]['save_count'] = 1
                        if should_save:
                            save_unknown_face(frame, cam_id)
                            last_unknown_ts = now
                    # واکنش صوتی
                    speak_tts_async("کاربر ناشناس شناسایی شد.") if in_working_hours() else speak_tts_async("هشدار: ورود غیرمجاز.")
                else:
                    seen_names.append(name)
                    with greet_lock:
                        if now - last_greet[name] > GREETING_INTERVAL:
                            speak_tts_async(f"{name} خوش آمدید.")
                            last_greet[name] = now
                    with presence_lock:
                        presence_log[name] = {"last_seen": now, "present": True, "cam_id": cam_id}

            # به‌روزرسانی حضور (خروج)
            with presence_lock:
                for tracked_name in list(presence_log.keys()):
                    if presence_log[tracked_name]["cam_id"] == cam_id:
                        if tracked_name not in seen_names:
                            if time.time() - presence_log[tracked_name]["last_seen"] > 10:
                                presence_log[tracked_name]["present"] = False

        time.sleep(0.01)

    cap.release()

# ------------- گزارش وضعیت ------------------
def report_room_status():
    print("\n======= آمار حضور افراد در {} =======".format(ROOM_NAME))
    with presence_lock:
        present = [n for n,i in presence_log.items() if i["present"]]
        absent = [n for n,i in presence_log.items() if not i["present"]]
        if present:
            for n in present:
                cam_id = presence_log[n]["cam_id"]
                last_seen = datetime.fromtimestamp(presence_log[n]["last_seen"]).strftime("%H:%M:%S")
                print(f"✅ {n} (دوربین {cam_id}, آخرین مشاهده: {last_seen})")
        else:
            print("✅ هیچ‌کس حاضر نیست")
        if absent:
            print(f"🚪 افراد خارج‌شده: {', '.join(absent)}")
        else:
            print("🚪 هیچ‌کس خارج نشده")
    print("================================")
    if unknown_faces_log:
        print("⚠️ چهره‌های ناشناس ثبت‌شده (10 مورد آخر):")
        for e in list(unknown_faces_log)[-10:]:
            print(f"  دوربین {e['cam_id']}, زمان: {e['time']}, فایل: {e['path']}")
    print("================================\n")

def cleanup_old_faces():
    now = time.time()
    if not os.path.exists(UNKNOWN_FOLDER):
        return
    try:
        for f in os.listdir(UNKNOWN_FOLDER):
            path = os.path.join(UNKNOWN_FOLDER, f)
            if os.path.isfile(path):
                if now - os.path.getmtime(path) > CLEANUP_HOURS * 3600:
                    os.remove(path)
                    logging.info(f"[CLEANUP] حذف شد: {f}")
    except Exception as e:
        logging.error(f"[CLEANUP] {e}")

# ------------- main ------------------
def main():
    os.makedirs(UNKNOWN_FOLDER, exist_ok=True)

    # Workerها
    threading.Thread(target=db_worker, daemon=True).start()
    threading.Thread(target=notify_worker, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=audio_worker, daemon=True).start()

    load_known_faces()

    for cam_id in CAMERA_IDS:
        threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()

    last_cleanup = 0.0
    while not shutdown_event.is_set():
        try:
            cleanup_old_faces()
            report_room_status()
            now = time.time()
            if now - last_cleanup > 24*3600:
                cleanup_unknown_faces_daily()
                last_cleanup = now
            time.sleep(300)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"[Main Loop] {e}")
            time.sleep(5)

    shutdown_event.set()
    time.sleep(1.0)

if __name__ == "__main__":
    main()
