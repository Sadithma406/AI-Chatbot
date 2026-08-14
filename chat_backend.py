import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODEL_PATH = "saved_model/intent_model.keras"
CLASSES_PATH = "saved_model/classes.json"
REPLIES_PATH = "saved_model/replies.json"

USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

CONFIDENCE_THRESHOLD = 0.25

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.get_logger().setLevel("ERROR")

print("Loading Universal Sentence Encoder...")

use = hub.load(USE_URL)

print("Universal Sentence Encoder loaded successfully.")

print("Loading intent classification model...")

model = tf.keras.models.load_model( MODEL_PATH,compile=False)

print("Intent model loaded successfully.")

with open(CLASSES_PATH,"r",encoding="utf-8") as f:
    intent_classes = json.load(f)


with open(REPLIES_PATH,"r",encoding="utf-8") as f:
    replies = json.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat(request: ChatRequest):
    user_input = request.message.strip().lower()
    # Empty message
    if not user_input:
        return {
            "intent": "unknown",
            "reply": "Please enter a message.",
            "confidence": 0
  }
    embedding = use([user_input])
    embedding = embedding.numpy().astype(np.float32)
    prediction = model.predict(embedding,verbose=0)
    predicted_index = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_index])
    predicted_intent = intent_classes[predicted_index]

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_intent = "unknown"
        reply = (
            "I'm sorry, I didn't quite understand that. "
            "Could you please rephrase?"
        )
    else:
        reply = replies.get(
            predicted_intent,
            {}
        ).get(
            "reply",
            "Sorry, I don't understand that."
        )
    return {
        "intent": predicted_intent,
        "reply": reply,
        "confidence": round(
            confidence * 100,
            2
        )
    }
app.mount("/",StaticFiles(directory="public",html=True),name="public")