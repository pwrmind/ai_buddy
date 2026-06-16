import os
import sys

# 1. Специфичный для Windows фикс путей CUDA (до импорта тяжелых библиотек)
if sys.platform == "win32":
    scripts_dir = os.path.dirname(sys.executable)
    venv_root = os.path.dirname(scripts_dir) 
    site_packages = os.path.join(venv_root, "Lib", "site-packages")
    
    cuda_paths = [
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "bin")
    ]
    
    current_path = os.environ.get("PATH", "")
    paths_to_add = []
    for path in cuda_paths:
        if os.path.exists(path):
            os.add_dll_directory(path)
            paths_to_add.append(path)
    if paths_to_add:
        os.environ["PATH"] = ";".join(paths_to_add) + ";" + current_path
        print("[СИСТЕМА]: Библиотеки CUDA успешно зарегистрированы в PATH процесса.")

# 2. Стандартные импорты
import io
import wave
import pyaudio
import pyttsx3
import ollama
from faster_whisper import WhisperModel

# --- НАСТРОЙКИ ---
MODEL_NAME = "gemma4:e4b-it-q4_K_M"

# Инициализация TTS
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "Russian" in voice.name or "ru" in voice.languages:
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 180)

# Инициализация Faster Whisper (теперь библиотеки гарантированно найдутся)
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

# Настройки аудиозахвата
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500  # Поднимите до 800+, если микрофон ловит эхо от колонок
SILENCE_LIMIT = 1.3      # Секунды тишины для отсечки фразы

def speak(text):
    print(f"ИИ: {text}")
    engine.say(text)
    engine.runAndWait()

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("\n[Слушаю...]")
    frames = []
    silent_chunks = 0
    has_spoken = False
    max_silent_chunks = int(SILENCE_LIMIT * RATE / CHUNK)

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        
        rms = wave.struct.unpack(f"{CHUNK}h", data)
        volume = max(abs(x) for x in rms)
        
        if volume > SILENCE_THRESHOLD:
            has_spoken = True
            silent_chunks = 0
        elif has_spoken:
            silent_chunks += 1
            if silent_chunks > max_silent_chunks:
                break
                
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio_buffer = io.BytesIO()
    wf = wave.open(audio_buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    audio_buffer.seek(0)
    return audio_buffer

def main():
    print(f"=== Голосовой чат запущен. Модель: {MODEL_NAME} ===")
    
    while True:
        audio_data = record_audio()
        
        print("[Распознаю речь...]")
        segments, _ = whisper_model.transcribe(audio_data, language="ru")
        user_text = "".join([segment.text for segment in segments]).strip()
        
        if not user_text:
            continue
            
        print(f"Вы: {user_text}")
        
        if any(word in user_text.lower() for word in ["выход", "стоп", "пока"]):
            speak("Отключаюсь. До связи!")
            break
            
        print("[Gemma думает...]")
        try:
            response = ollama.generate(
                model=MODEL_NAME, 
                prompt=user_text,
                system="Ты — краткий и емкий голосовой ассистент. Отвечай только текстом на русском языке, понятным для чтения вслух. Категорически не используй Markdown разметку, списки, формулы, звездочки (*) или решетки (#)."
            )
            ai_text = response['response']
            speak(ai_text)
            
        except Exception as e:
            print(f"Ошибка Ollama: {e}")

if __name__ == "__main__":
    main()
