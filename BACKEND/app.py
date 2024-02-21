import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.optimizers import Adam
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load your dataset
df = pd.read_csv("dataset.csv")

# Define maximum sequence length and vocabulary size
MAX_SEQ_LENGTH = 100
MAX_VOCAB_SIZE = 10000

# Tokenize text data
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df['Keywords'])
sequences = tokenizer.texts_to_sequences(df['Keywords'])

# Pad sequences to ensure uniform length
X = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Interest'])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=100, input_length=MAX_SEQ_LENGTH),
    LSTM(128),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Retrieve input data from the request
        user_description = request.json.get('interest', '')

        if not user_description:
            return jsonify([])

        # Tokenize and pad user input
        user_sequence = tokenizer.texts_to_sequences([user_description])
        user_padded_sequence = pad_sequences(user_sequence, maxlen=MAX_SEQ_LENGTH)

        # Predict probabilities for user input using the trained LSTM model
        probabilities = model.predict(user_padded_sequence)[0]

        # Get top 5 recommendations
        top_n_indices = np.argsort(probabilities)[-5:][::-1]
        recommendations = [label_encoder.classes_[idx] for idx in top_n_indices]

        print("-----------------------")
        print(recommendations)
        print("-----------------------")
        return jsonify(recommendations)

    except Exception as e:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)