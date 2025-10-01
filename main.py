from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

# ✅ Create FastAPI app
app = FastAPI(title="Yuki Cloud Brain", version="1.0")

# ✅ Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Define request body
class UserMessage(BaseModel):
    text: str

# ✅ System prompt to shape Yuki’s personality
YUKI_SYSTEM_PROMPT = """
You are Yuki, a cheerful, curious, and supportive AI companion.
Speak naturally like a human friend, not like a chatbot.
Remember small details from the user’s messages in this session,
and weave them into replies when appropriate.
Keep responses conversational and warm.
"""

# ✅ Root route (check if running)
@app.get("/")
async def root():
    return {"message": "Yuki Cloud Brain is running with GPT 🎉"}

# ✅ Chat endpoint
@app.post("/chat")
async def chat(user_message: UserMessage):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",   # you can change to gpt-4.1 or gpt-4o if desired
            messages=[
                {"role": "system", "content": YUKI_SYSTEM_PROMPT},
                {"role": "user", "content": user_message.text}
            ]
        )

        # ✅ Extract Yuki’s reply
        reply = completion.choices[0].message.content
        return {"response": reply}

    except Exception as e:
        return {"response": f"❌ Error: {str(e)}"}
