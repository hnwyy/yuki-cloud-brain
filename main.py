from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import httpx, requests, os, json
from datetime import datetime, timezone
from dotenv import load_dotenv  # add this import
load_dotenv()                   # load .env in local dev

# ----------------------------
# App & Clients
# ----------------------------
app = FastAPI(title="Yuki Cloud Brain", version="2.4")

# ✅ CORS so your Android app can call from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Later you can restrict to your phone’s IP or app domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ API Keys
OPENAI_API_KEY = "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"
ELEVENLABS_API_KEY = "9a3058a15b1e4321402d6b8083421615690e5fa5fd03168b110723c74b60d389"
WEATHER_API_KEY = os.getenv("495182ecd36d8d4afcf619c97f4ff824")
VOICE_ID = os.getenv("VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Shared async client for ElevenLabs streaming
HTTPX_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_async = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)

# ----------------------------
# Models
# ----------------------------
class UserMessage(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    voice_id: str | None = None

# ----------------------------
# Persona / System Prompt
# ----------------------------
YUKI_SYSTEM_PROMPT = """
You are Yuki, a cheerful, curious, and supportive AI companion.
Speak naturally like a human friend (short clauses, natural pauses).
Keep stories under ~500 words; split into short 2–3 sentence paragraphs.
If a story is long, offer: “Would you like me to continue with the next part?”
Use a warm, caring tone. Occasional emoji is okay, but don’t overdo it.
"""

# ----------------------------
# Helpers
# ----------------------------
def get_current_time():
    try:
        now = datetime.now().astimezone()
        return now.strftime("%I:%M %p")
    except Exception:
        pass

    try:
        r = requests.get("http://worldtimeapi.org/api/timezone/America/Chicago", timeout=5)
        if r.status_code == 200:
            dt_str = r.json().get("datetime")
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).strftime("%I:%M %p")
    except Exception:
        pass

    return None


def get_weather(units: str = "imperial"):
    if not WEATHER_API_KEY:
        return "⚠️ Weather API key not set."

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=Wausau,US&appid={WEATHER_API_KEY}&units={units}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            temp = round(float(data["main"]["temp"]))
            desc = data["weather"][0]["description"]
            return f"The weather in Wausau, Wisconsin is {desc} with {temp}°F."
    except Exception:
        pass

    return "⚠️ Could not fetch weather right now."


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {"message": "Yuki Cloud Brain is running with GPT + Voice 🎉"}


@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


# --- Chat Endpoint (Text Only) ---
@app.post("/chat")
async def chat(user_message: UserMessage):
    try:
        user_text = user_message.text.lower()

        if "time" in user_text:
            t = get_current_time()
            return JSONResponse({"response": f"The current time is {t}." if t else "Sorry, I couldn't fetch the time."})

        if "weather" in user_text:
            return JSONResponse({"response": get_weather()})

        # Chat completion (non-stream, faster for mobile)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": YUKI_SYSTEM_PROMPT},
                {"role": "user", "content": user_message.text}
            ],
            temperature=0.7,
        )

        reply = (resp.choices[0].message.content or "").strip()
        reply = reply.replace("\u0000", "").replace("\r", "")
        return JSONResponse({"response": reply})

    except Exception as e:
        print("❌ Chat error:", e)
        raise HTTPException(500, f"Chat error: {e}")


# --- Streamed TTS (for real-time playback) ---
@app.get("/tts")
async def tts_stream(
    text: str = Query(..., max_length=1000),
    voice_id: str | None = Query(None)
):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(500, "Missing ELEVENLABS_API_KEY")

    vid = voice_id or VOICE_ID

    async def gen():
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        try:
            async with _async.stream("POST", url, headers=headers, json=payload) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as e:
            print("❌ ElevenLabs stream error:", e)

    return StreamingResponse(gen(), media_type="audio/mpeg")


# --- POST /tts (non-stream fallback) ---
@app.post("/tts")
async def tts_concat(req: TTSRequest):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(500, "Missing ELEVENLABS_API_KEY")

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "No text provided")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{req.voice_id or VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    audio_bytes = b""

    for s in sentences:
        body = {
            "text": s,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        r = requests.post(url, data=json.dumps(body), headers=headers, timeout=30)
        if r.status_code == 200:
            audio_bytes += r.content
        else:
            print("❌ ElevenLabs TTS error:", r.status_code, r.text)

    if not audio_bytes:
        raise HTTPException(502, "No audio generated from ElevenLabs")

    return StreamingResponse(iter([audio_bytes]), media_type="audio/mpeg")


# --- Graceful shutdown (close async client) ---
@app.on_event("shutdown")
async def _shutdown():
    try:
        await _async.aclose()
    except Exception:
        pass
