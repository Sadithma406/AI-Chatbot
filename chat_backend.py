import sys
import json
import numpy as np
import tensorflow as tf
import os

# Hide TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

model = tf.keras.models.load_model("saved_model/intent_model.keras")

with open("saved_model/classes.json", "r") as f:
    intent_classes = json.load(f)

with open("saved_model/replies.json", "r") as f:
    replies = json.load(f)

# Tell Node.js we're ready
print(json.dumps({"status": "ready"}), flush=True)

while True:
    try:
        user_input = input()

        prediction = model.predict(
            tf.constant([user_input.lower()]),
            verbose=0
        )

        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_index])

        predicted_intent = intent_classes[predicted_index]

        # Add a confidence threshold for random inputs
        CONFIDENCE_THRESHOLD = 0.5
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

        result = {
            "intent": predicted_intent,
            "reply": reply,
            "confidence": confidence
        }

        print(json.dumps(result), flush=True)

    except Exception as e:
        print(json.dumps({
            "error": str(e)
        }), flush=True)