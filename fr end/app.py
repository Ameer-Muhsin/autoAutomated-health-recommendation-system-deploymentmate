


from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


def custom_tokenizer(text):
    return text.split()

# Load model and utilities
model = load_model("model.h5")
with open("Tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("encoder.pkl", "rb") as f:
     label_encoder = pickle.load(f)
with open("main project.json", "r") as f:
    intents = json.load(f)
num_words = 1000  # Vocabulary size
max_len = 20 


file_name = "main project.json"
with open(file_name, "r") as file:
    intents = json.load(file)

# Preprocess data
patterns = []
labels = []
tag_responses = {}

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])
    tag_responses[intent["tag"]] = intent["responses"]

def chatbot_response(input_text):
    print(type(input_text))
    sequence = tokenizer.transform([input_text]).toarray()
    # padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")
    prediction = model.predict(sequence, verbose=0)
    intent_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([intent_index])[0]
    print(predicted_label,"---------------------")
    return np.random.choice(tag_responses[predicted_label])




# Predict the intent
def predict_intent(user_input):
    input_vector = vectorizer.transform([user_input]).toarray()
    prediction = model.predict(input_vector)
    intent_index = np.argmax(prediction)
    intent_name = label_encoder.inverse_transform([intent_index])[0]
    return intent_name

# Get a response
def get_response(intent_name):
    for intent in intents["tag"]:
        if intent["tag"] == intent_name:
            return np.random.choice(intent["responses"])
        return "Sorry, I didn't understand that."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    print("----------------------------",user_input)
    response = chatbot_response(user_input)
    print("**********************")
    # response = get_response(intent)
    return jsonify({"response": response})
