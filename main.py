import os
import sys
import re

# 1. Специфичный для Windows фикс путей CUDA (выполняется до импорта тяжелых библиотек)
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

# 2. Основные импорты
import io
import wave
import pyaudio
import pyttsx3
import ollama
from faster_whisper import WhisperModel

# --- НАСТРОЙКИ ---
MODEL_NAME = "gemma4:e4b-it-q4_K_M"

# 3. Инициализация Faster Whisper на GPU NVIDIA
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

# Настройки аудиозахвата для режима Hands-free
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500  # Чувствительность к шуму.
SILENCE_LIMIT = 1.3      # Секунды тишины, после которых фраза считается законченной

def speak(text):
    """Вывод текста в консоль и надежная озвучка с переинициализацией движка"""
    print(f"ИИ: {text}")
    
    try:
        engine = pyttsx3.init(driverName='sapi5') 
    except Exception:
        engine = pyttsx3.init()
        
    voices = engine.getProperty('voices')
    for voice in voices:
        if "Russian" in voice.name or "ru" in voice.languages or "IRINA" in voice.id.upper() or "PAVEL" in voice.id.upper():
            engine.setProperty('voice', voice.id)
            break
            
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1.0)
    
    engine.say(text)
    engine.runAndWait()
    del engine

def record_audio():
    """Запись звука с микрофона. Автоматический стоп, когда пользователь замолчал"""
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
    
    # Инициализация истории диалога с системным промптом
    messages = [
        {
            "role": "system",
            "content": "Ты — краткий, чуткий и емкий голосовой ассистент. Отвечай только текстом на русском языке, понятным для чтения вслух. Категорически не используй Markdown разметку, списки, формулы, звездочки (*) или решетки (#). Твои ответы должны быть не длиннее 2-3 предложений, чтобы их было комфортно слушать."
        }
    ]
    
    while True:
        # Шаг 1: Запись фразы
        audio_data = record_audio()
        
        # Шаг 2: Расшифровка на видеокарте через CUDA
        print("[Распознаю речь...]")
        segments, _ = whisper_model.transcribe(audio_data, language="ru")
        user_text = "".join([segment.text for segment in segments]).strip()
        
        if not user_text:
            continue
            
        print(f"Вы: {user_text}")
        
        # Умная проверка стоп-слов с помощью регулярных выражений (\b защищает от ложных срабатываний)
        is_exit_command = False
        for word in ["выход", "стоп", "пока"]:
            if re.search(r'\b' + re.escape(word) + r'\b', user_text.lower()):
                # Игнорируем «пока», если дальше идут уточняющие слова (пока я, пока что и т.д.)
                if word == "пока" and re.search(r'\bпока\s+(что|я|мы|ты|вы|не|хочу|можно|давай)\b', user_text.lower()):
                    continue
                is_exit_command = True
                break

        if is_exit_command:
            speak("Отключаюсь. До связи!")
            break
            
        # Добавляем реплику пользователя в память диалога
        messages.append({"role": "user", "content": user_text})
        
        # Шаг 3: Запрос к Gemma 4 через ollama.chat с передачей ВСЕЙ истории
        print("[Gemma думает...]")
        try:
            response = ollama.chat(
                model=MODEL_NAME, 
                messages=messages
            )
            ai_text = response['message']['content']
            
            # Добавляем ответ ИИ в память диалога для сохранения контекста
            messages.append({"role": "assistant", "content": ai_text})
            
            # Шаг 4: Озвучка ответа
            speak(ai_text)
            
            # Ограничиваем длину истории (опционально), чтобы не перегружать контекст бесконечно
            if len(messages) > 21:  # Храним системный промпт + последние 10 раундов диалога
                messages = [messages[0]] + messages[-20:]
            
        except Exception as e:
            print(f"Ошибка Ollama: {e}")

if __name__ == "__main__":
    main()
