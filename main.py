from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import requests, os, base64

# ✅ Create FastAPI app
app = FastAPI(title="Yuki Cloud Brain", version="1.1")

# ✅ Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ ElevenLabs setup
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # default voice ID (you can change later)

# ✅ Request model
class UserMessage(BaseModel):
    text: str

YUKI_SYSTEM_PROMPT = """
You are Yuki, a cheerful, curious, and supportive AI companion.
Speak naturally like a human friend, not like a chatbot.
"""

@app.get("/")
async def root():
    return {"message": "Yuki Cloud Brain is running with GPT + Voice 🎉"}

def text_to_speech(text: str) -> str:
    """Convert text to base64 audio using ElevenLabs."""
    if not ELEVENLABS_API_KEY:
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }

    response = requests.post(url, json=body, headers=headers)
    if response.status_code != 200:
        return None

    # Convert audio bytes → base64 string
    return base64.b64encode(response.content).decode("utf-8")

@app.post("/chat")
async def chat(user_message: UserMessage):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": YUKI_SYSTEM_PROMPT},
                {"role": "user", "content": user_message.text}
            ]
        )
        reply = completion.choices[0].message.content

        # ✅ Generate TTS audio
        audio_b64 = text_to_speech(reply)

        return JSONResponse(content={
            "response": reply,
            "audio": audio_b64
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)})
