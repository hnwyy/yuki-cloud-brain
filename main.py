from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import requests, os, base64
from datetime import datetime

# ✅ Create FastAPI app
app = FastAPI(title="Yuki Cloud Brain", version="1.3")

# ✅ Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ ElevenLabs setup
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # replace with your ElevenLabs voice ID

# ✅ OpenWeather setup
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # set this in Render

# ✅ Request model
class UserMessage(BaseModel):
    text: str

# ✅ System prompt
YUKI_SYSTEM_PROMPT = """
You are Yuki, a cheerful, curious, and supportive AI companion.
Speak naturally like a human friend, not like a chatbot.
When telling stories, split them into short paragraphs of 2–3 sentences each,
as if pausing between chapters so they can be spoken smoothly.
"""

@app.get("/")
async def root():
    return {"message": "Yuki Cloud Brain is running with GPT + Voice 🎉"}

# ✅ Time API
def get_current_time():
    try:
        response = requests.get("http://worldtimeapi.org/api/ip")
        if response.status_code == 200:
            data = response.json()
            dt_str = data["datetime"]
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.strftime("%I:%M %p")  # e.g., 03:45 PM
    except Exception as e:
        print("❌ Time API failed:", str(e))
    return None

# ✅ Weather API (always Wausau, WI)
def get_weather(units: str = "imperial"):
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    if not WEATHER_API_KEY:
        return "⚠️ Weather API key not set"

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=Wausau,US&appid={WEATHER_API_KEY}&units={units}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"The weather in Wausau, Wisconsin is {desc} with {temp}°F."
    except Exception as e:
        print("❌ Weather API failed:", str(e))
    return "⚠️ Could not fetch weather right now."

# ✅ TTS with chunking
def text_to_speech_chunks(text: str) -> list[str]:
    if not ELEVENLABS_API_KEY:
        print("⚠️ ElevenLabs API key missing")
        return []

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    chunks = [c.strip() for c in text.split(". ") if c.strip()]
    audio_clips = []

    for i, chunk in enumerate(chunks, start=1):
        body = {
            "text": chunk,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }

        try:
            response = requests.post(url, json=body, headers=headers, timeout=30)
            print(f"🔊 ElevenLabs chunk {i}/{len(chunks)}: {response.status_code}")

            if response.status_code == 200:
                audio_clips.append(base64.b64encode(response.content).decode("utf-8"))
            else:
                print("❌ ElevenLabs error:", response.text)
        except Exception as e:
            print(f"❌ ElevenLabs request failed on chunk {i}:", str(e))

    return audio_clips

# ✅ Chat endpoint
@app.post("/chat")
async def chat(user_message: UserMessage):
    try:
        user_text = user_message.text.lower()

        # Time
        if "time" in user_text:
            current_time = get_current_time()
            reply = f"The current time is {current_time}" if current_time else "Sorry, I couldn't fetch the time."
            audio_clips = text_to_speech_chunks(reply)
            return JSONResponse(content={"response": reply, "audio": audio_clips})

        # Weather (always Wausau)
        if "weather" in user_text:
            reply = get_weather()
            audio_clips = text_to_speech_chunks(reply)
            return JSONResponse(content={"response": reply, "audio": audio_clips})

        # GPT default
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": YUKI_SYSTEM_PROMPT},
                {"role": "user", "content": user_message.text}
            ]
        )
        reply = completion.choices[0].message.content
        audio_clips = text_to_speech_chunks(reply)

        return JSONResponse(content={"response": reply, "audio": audio_clips})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})
