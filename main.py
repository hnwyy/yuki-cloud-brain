from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import requests, os
from datetime import datetime

# ✅ Create FastAPI app
app = FastAPI(title="Yuki Cloud Brain", version="2.1")

# ✅ Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ ElevenLabs setup
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # replace with your chosen ElevenLabs voice

# ✅ OpenWeather setup
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# ✅ Request models
class UserMessage(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str

# ✅ System prompt
YUKI_SYSTEM_PROMPT = """
You are Yuki, a cheerful, curious, and supportive AI companion.
Speak naturally like a human friend, not like a chatbot.
When telling stories, keep them under 500 words, and split into short
paragraphs of 2–3 sentences each, as if pausing between chapters so they can be spoken smoothly.
If the story is long, break it into multiple parts and say something like,
‘Would you like me to continue with the next part?’
"""

@app.get("/")
async def root():
    return {"message": "Yuki Cloud Brain is running with GPT + Voice 🎉"}

# ✅ Time API
def get_current_time():
    try:
        response = requests.get("http://worldtimeapi.org/api/timezone/America/Chicago")
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

# ✅ Chat endpoint (returns text only)
@app.post("/chat")
async def chat(user_message: UserMessage):
    try:
        user_text = user_message.text.lower()

        # Special cases
        if "time" in user_text:
            current_time = get_current_time()
            reply = f"The current time is {current_time}" if current_time else "Sorry, I couldn't fetch the time."
            return JSONResponse(content={"response": reply})

        if "weather" in user_text:
            reply = get_weather()
            return JSONResponse(content={"response": reply})

        # ✅ GPT streaming (OpenAI 2.x SDK)
        full_text = []
        with client.chat.completions.stream(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": YUKI_SYSTEM_PROMPT},
                {"role": "user", "content": user_message.text}
            ]
        ) as stream:
            for event in stream:
                if event.type == "token":
                    full_text.append(event.token)

        reply = "".join(full_text)
        return JSONResponse(content={"response": reply})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# ✅ TTS endpoint (convert text → audio file, then stream back)
@app.post("/tts")
async def tts(req: TTSRequest):
    text = req.text
    if not text:
        return JSONResponse(content={"error": "No text provided"})

    print(f"🔊 Sending text to ElevenLabs: {text}")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    body = {
        "text": text,
        "model_id": "eleven_multilingual_v2",  # more stable than flash
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }

    r = requests.post(url, json=body, headers=headers)
    if r.status_code != 200:
        print("❌ ElevenLabs error:", r.text)
        return JSONResponse(content={"error": r.text})

    # Send full MP3 response to client
    return StreamingResponse(iter([r.content]), media_type="audio/mpeg")