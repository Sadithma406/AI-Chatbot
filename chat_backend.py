import json
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles


USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

CONFIDENCE_THRESHOLD = 0.15

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

print("Loading Universal Sentence Encoder...")
use = hub.load(USE_URL)
print("Universal Sentence Encoder loaded successfully.")

print("Loading intent classification model...")
model = tf.keras.models.load_model("saved_model/intent_model.keras", compile=False)
print("Intent model loaded successfully.")

# Load answers map (index -> list of answers)
with open("saved_model/answers.json", "r", encoding="utf-8") as f:
    answers_map = json.load(f)

app = FastAPI()


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = str(data.get("message", "")).strip().lower()

    embedding = use([user_input])
    embedding = embedding.numpy().astype(np.float32)
    prediction = model.predict(embedding, verbose=0)
    predicted_index = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_index])

    if confidence < CONFIDENCE_THRESHOLD or str(predicted_index) not in answers_map:
        reply = (
            "I'm sorry, I didn't quite understand that. "
            "Could you please rephrase?"
        )
    else:
        answers = answers_map[str(predicted_index)]
        reply = random.choice(answers)

    return {
        "reply": reply,
    }

app.mount("/", StaticFiles(directory="public", html=True), name="public")

