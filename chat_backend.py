import json
import numpy as np
import tensorflow as tf
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Hide TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Load TensorFlow model only once
model = tf.keras.models.load_model("saved_model/intent_model.keras")

with open("saved_model/classes.json", "r") as f:
    intent_classes = json.load(f)

with open("saved_model/replies.json", "r") as f:
    replies = json.load(f)

print("TensorFlow model loaded successfully.")

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body
class ChatRequest(BaseModel):
    message: str

# Chat endpoint
@app.post("/api/chat")
def chat(request: ChatRequest):

    user_input = request.message.strip().lower()

    prediction = model.predict(
        tf.constant([user_input]),
        verbose=0
    )

    predicted_index = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_index])

    predicted_intent = intent_classes[predicted_index]

    CONFIDENCE_THRESHOLD = 0.50

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_intent = "unknown"
        reply = "I'm sorry, I didn't quite understand that. Could you please rephrase?"
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
        "confidence": round(confidence * 100, 2)
    }
# Serve static files
app.mount("/", StaticFiles(directory="public", html=True), name="public")
