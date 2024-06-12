import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gradio as gr

# Load the saved RNN model
model_rnn = tf.keras.models.load_model("rnn_model.h5")

# Load tokenizer (assumes you have saved it previously)
import pickle
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define total_words (must match the number used during training)
total_words = len(tokenizer.word_index) + 1
max_sequence_len = 100  # Must match the length used during training

def generate_text(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model_rnn.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        
        seed_text += " " + output_word
    
    return seed_text

# Gradio Interface
def predict(input_text):
    return generate_text(input_text)

interface = gr.Interface(fn=predict, 
                         inputs="text", 
                         outputs="text",
                         title="Sherlock Holmes Text Generator",
                         description="Generate text based on Sherlock Holmes stories using an RNN model.")

interface.launch()
