from os import name
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
pickle_open = open("tokenizer.pkl","rb")
tokenizer = pickle.load(pickle_open)
model = load_model("model.h5")


def predict(t):
    example = tokenizer.texts_to_sequences([t])
    example = pad_sequences(example, maxlen=2)
    prediction = model.predict(np.array(example))
    predicted_word = np.argmax(prediction)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))  # https://stackoverflow.com/a/43927939/246508
    return reverse_word_map[predicted_word]
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/after", methods=["POST"])
def after():
    text =(json.loads(request.data))
    finaltext = text["text"]
    print(finaltext)
    next = predict(f"{finaltext}")
    return f"<h1>{next}</h1>"

if __name__ == '__main__':
    app.run(debug=True)
