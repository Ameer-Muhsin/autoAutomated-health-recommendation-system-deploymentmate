import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Load intents
with open("main project.json", "r") as f:
    intents = json.load(f)

# Prepare data
texts = []
labels = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["intents"])

# Convert text to bag-of-words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Save vectorizer and label encoder
with open("Tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(set(labels)), activation="softmax"))

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    metrics=["accuracy"],
)

# Train the model
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# Save the model
model.save("model.h5")
print("Model trained and saved!")
