import os
import asyncio
from io import BytesIO
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import whisper
from gtts import gTTS
from pydub import AudioSegment
from groq import Groq

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set ffmpeg path for pydub
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"  # Ensure this path is correct

# Initialize Whisper (tiny for speed)
model = whisper.load_model("tiny")

# Initialize Groq
client = Groq(api_key=GROQ_API_KEY)

# Conversation memory per user
user_context = {}

# -----------------------------
# COMMAND HANDLER
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ഹലോ! ഞാൻ നിങ്ങളുടെ കൃഷി സഹായി.\n"
        "Malayalam അല്ലെങ്കിൽ English വോയ്സ് / ടെക്സ്റ്റ് മെസേജ് അയയ്ക്കൂ."
    )

# -----------------------------
# TEXT & VOICE HANDLER
# -----------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_context.setdefault(user_id, [])

    try:
        # -----------------------------
        # 1️⃣ Get user input
        # -----------------------------
        if update.message.text:
            user_text = update.message.text.strip()
            detected_lang = "en"  # Assume text is English if not detected
        elif update.message.voice:
            voice_file = await update.message.voice.get_file()
            voice_bytes = await voice_file.download_as_bytearray()
            audio_buffer = BytesIO(voice_bytes)

            # Convert OGG -> WAV in-memory
            wav_buffer = await asyncio.to_thread(AudioSegment.from_file, audio_buffer, "ogg")
            wav_io = BytesIO()
            await asyncio.to_thread(wav_buffer.export, wav_io, format="wav")
            wav_io.seek(0)

            # Transcribe using Whisper
            result = await asyncio.to_thread(model.transcribe, wav_io)
            user_text = result["text"].strip()
            detected_lang = result.get("language", "en")
        else:
            await update.message.reply_text("Only text or voice messages are supported.")
            return

        print(f"[{user_id}] User input ({detected_lang}): {user_text}")
        user_context[user_id].append({"role": "user", "content": user_text})
        context_history = user_context[user_id][-5:]  # Keep last 5 messages

        # -----------------------------
        # 2️⃣ Prepare prompt for Groq
        # -----------------------------
        if detected_lang.lower().startswith("en"):
            prompt = (
                f"You are a Malayalam-speaking farming assistant. "
                f"The user said in English: {user_text}\n"
                f"Reply in Malayalam. Include the English message and Malayalam translation in your reply."
            )
        else:
            prompt = (
                f"You are a Malayalam-speaking farming assistant. "
                f"The user said in Malayalam: {user_text}\n"
                f"Reply naturally in Malayalam."
            )

        # -----------------------------
        # 3️⃣ Call Groq API
        # -----------------------------
        chat_completion = client.chat.completions.create(
            messages=context_history + [{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant"
        )
        answer = chat_completion.choices[0].message.content.strip()
        print(f"[{user_id}] AI answer: {answer}")

        user_context[user_id].append({"role": "assistant", "content": answer})

        # -----------------------------
        # 4️⃣ Send text reply
        # -----------------------------
        await update.message.reply_text(answer)

        # -----------------------------
        # 5️⃣ Convert to Malayalam speech (TTS)
        # -----------------------------
        tts_buffer = BytesIO()
        await asyncio.to_thread(lambda: gTTS(text=answer, lang="ml").write_to_fp(tts_buffer))
        tts_buffer.seek(0)
        await update.message.reply_audio(tts_buffer)

    except Exception as e:
        print(f"[{user_id}] Error: {e}")
        await update.message.reply_text(
            "ക്ഷമിക്കണം! ഒരു പിശക് സംഭവിച്ചു, വീണ്ടും ശ്രമിക്കുക."
        )

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT | filters.VOICE, handle_message))
    print("Fast Malayalam bot started...")
    app.run_polling()
