# -*- coding: utf-8 -*-
"""
Multi-Class Text Emotion Classification using BERT

This script builds a neural network that can classify text into 7 different emotions:
anger, disgust, fear, happy, neutral, sad, surprise

The model uses BERT (Bidirectional Encoder Representations from Transformers),
a pre-trained language model, as the foundation for understanding text.
"""

# Import necessary libraries
import pandas as pd        # For data manipulation and analysis (creating DataFrames)
import numpy as np         # For numerical operations and array handling
import tensorflow as tf    # Deep learning framework for building and training models
from transformers import BertTokenizer, TFBertModel  # Hugging Face transformers for BERT
import os                  # For operating system interface (file/directory operations)
import glob                # For finding files matching specific patterns
from sklearn.model_selection import train_test_split  # For splitting data (not used in this version)
from tqdm.auto import tqdm # For progress bars during data processing
import zipfile             # For handling zip file extraction
from google.colab import drive  # For mounting Google Drive in Colab environment

"""
STEP 1: Initialize the BERT tokenizer
The tokenizer converts text into numbers that BERT can understand
"""
# Load the pre-trained BERT tokenizer
# 'bert-base-cased' means it distinguishes between uppercase and lowercase letters
# The tokenizer breaks text into smaller pieces (tokens) and converts them to numbers
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

"""
STEP 2: Load and prepare the emotion dataset from a zip file
"""
def load_emotion_dataset(zip_path, extract_path='/content/drive/MyDrive/EmotionClassText/'):
    """
    This function extracts emotion data from a zip file and organizes it into a DataFrame
    
    Parameters:
    - zip_path: Path to the zip file containing emotion data
    - extract_path: Where to extract the files
    
    Returns:
    - df: DataFrame containing text phrases and their emotion labels
    - emotions: List of emotion categories
    """
    
    # Check if the extraction folder already exists to avoid re-extracting
    if not os.path.exists(extract_path):
        print(f"Extracting zip file to {extract_path}...")
        # Open the zip file and extract all contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        print(f"Extraction folder {extract_path} already exists, skipping extraction.")

    # Define the 7 emotion categories we want to classify
    emotions = ['surprise', 'sad', 'neutral', 'happy', 'fear', 'disgust', 'anger']
    
    # Create an empty list to store all text-emotion pairs
    data = []

    # Loop through each emotion category
    for emotion in emotions:
        # Create the path to the folder containing texts for this emotion
        folder_path = os.path.join(extract_path, emotion)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Folder for {emotion} not found in extracted zip")
            continue
            
        # Find all .txt files in this emotion's folder
        files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        # Read each text file and add it to our data
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read().strip()  # Read and remove whitespace
                # Add this text-emotion pair to our data list
                data.append({'Phrase': text, 'Emotion': emotion})

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Create a mapping from emotion names to numbers (0-6)
    # Machine learning models work with numbers, not text labels
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
    
    # Add a numerical label column to the DataFrame
    df['Label'] = df['Emotion'].map(emotion_to_idx)
    
    return df, emotions

"""
STEP 3: Mount Google Drive and load the dataset
"""
# Mount Google Drive to access files stored there
drive.mount('/content/drive')

# Path to your emotion dataset zip file
zip_path = '/content/drive/MyDrive/EmotionClassText.zip'  # Update this path as needed

# Load the dataset using our function
df, classes = load_emotion_dataset(zip_path)

# Display information about our dataset
df.info()    # Shows data types, non-null counts, memory usage
df.head()    # Shows first 5 rows of data
df['Emotion'].value_counts()  # Shows how many examples we have for each emotion

"""
STEP 4: Data preprocessing and tokenization functions
"""
def prepare_data(text, tokenizer):
    """
    This function converts a single text string into BERT-compatible format
    
    Parameters:
    - text: The input text string
    - tokenizer: BERT tokenizer object
    
    Returns:
    - Dictionary containing input_ids and attention_mask
    """
    
    # Use the tokenizer to process the text
    token = tokenizer.encode_plus(
        text,                    # The text to tokenize
        max_length=256,         # Maximum sequence length (longer texts get truncated)
        truncation=True,        # Cut off text if it's longer than max_length
        padding='max_length',   # Pad shorter texts with zeros to reach max_length
        add_special_tokens=True, # Add [CLS] at start and [SEP] at end (BERT requirements)
        return_tensors='tf'     # Return TensorFlow tensors
    )
    
    # Return the processed tokens in the format our model expects
    return {
        'input_ids': tf.cast(token.input_ids, tf.int32),      # The actual token numbers
        'attention_mask': tf.cast(token.attention_mask, tf.int32)  # Mask to ignore padding
    }

def generate_training_data(df, tokenizer):
    """
    This function processes all texts in the DataFrame for training
    
    Parameters:
    - df: DataFrame containing texts and labels
    - tokenizer: BERT tokenizer
    
    Returns:
    - input_ids: Array of tokenized texts
    - attn_masks: Array of attention masks
    - labels: One-hot encoded emotion labels
    """
    
    # Create arrays to store processed data
    # Each text becomes a sequence of 256 numbers (tokens)
    input_ids = np.zeros((len(df), 256), dtype=np.int32)
    attn_masks = np.zeros((len(df), 256), dtype=np.int32)

    # Process each text in the DataFrame
    for i, text in tqdm(enumerate(df['Phrase']), total=len(df)):
        # Tokenize this text
        tokenized = prepare_data(text, tokenizer)
        
        # Store the tokenized data in our arrays
        input_ids[i, :] = tokenized['input_ids']
        attn_masks[i, :] = tokenized['attention_mask']

    # Convert emotion labels to one-hot encoding
    # Instead of label=2 for 'neutral', we get [0,0,1,0,0,0,0]
    labels = np.zeros((len(df), 7), dtype=np.int32)  # 7 emotions = 7 columns
    # Set the correct position to 1 for each example
    labels[np.arange(len(df)), df['Label'].values] = 1
    
    return input_ids, attn_masks, labels

"""
STEP 5: Process all our data
"""
print("Processing all texts for training...")
input_ids, attn_masks, labels = generate_training_data(df, tokenizer)

"""
STEP 6: Create TensorFlow dataset for efficient training
"""
# Create a TensorFlow dataset from our processed data
# This allows for efficient batching and shuffling during training
dataset = tf.data.Dataset.from_tensor_slices((input_ids, attn_masks, labels))

# Reorganize the data structure to match what our model expects
# The model expects inputs as a dictionary with 'input_ids' and 'attention_mask' keys
dataset = dataset.map(lambda x, y, z: ({'input_ids': x, 'attention_mask': y}, z))

# Shuffle the data randomly and create batches of 16 examples
# Shuffling prevents the model from learning the order of examples
# Batching processes multiple examples at once for efficiency
dataset = dataset.shuffle(10000).batch(16, drop_remainder=True)

"""
STEP 7: Split data into training and validation sets
"""
# Use 80% of data for training, 20% for validation
p = 0.8
# Calculate how many batches to use for training
train_size = int((len(df)//16) * p)  # len(df)//16 = total number of batches

# Split the dataset
train_dataset = dataset.take(train_size)    # First 80% of batches for training
val_dataset = dataset.skip(train_size)      # Remaining 20% for validation

"""
STEP 8: Build the neural network model
"""
def build_model():
    """
    This function creates our emotion classification model using BERT
    
    The model has three main parts:
    1. BERT encoder (pre-trained) - understands the text
    2. Dense layer (512 neurons) - processes BERT's output
    3. Output layer (7 neurons) - predicts emotion probabilities
    """
    
    # Define input layers
    # These specify the shape and type of data the model expects
    input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
    attn_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

    # Load pre-trained BERT model
    # This model already knows a lot about language from training on billions of words
    bert = TFBertModel.from_pretrained('bert-base-cased')
    
    # Pass our inputs through BERT
    # [1] gets the pooled output (a single vector representing the whole text)
    # [0] would give us token-level representations
    bert_embds = bert(input_ids, attention_mask=attn_masks)[1]
    
    # Add a dense (fully connected) layer
    # This processes BERT's output and learns emotion-specific patterns
    # 512 neurons with ReLU activation (outputs 0 for negative inputs, input value for positive)
    intermediate = tf.keras.layers.Dense(512, activation='relu', name='intermediate')(bert_embds)
    
    # Output layer with 7 neurons (one for each emotion)
    # Softmax activation ensures outputs sum to 1 (interpretable as probabilities)
    output = tf.keras.layers.Dense(7, activation='softmax', name='output')(intermediate)

    # Create the complete model
    model = tf.keras.Model(inputs=[input_ids, attn_masks], outputs=output)
    
    # Configure the model for training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Adam optimizer with small learning rate
        loss=tf.keras.losses.CategoricalCrossentropy(),          # Loss function for multi-class classification
        metrics=[tf.keras.metrics.CategoricalAccuracy('accuracy')]  # Track accuracy during training
    )
    
    return model

"""
STEP 9: Create and train the model
"""
print("Building the model...")
model = build_model()

print("Starting training...")
# Train the model for 5 epochs (5 complete passes through the training data)
# Each epoch, the model sees all training examples once and updates its weights
history = model.fit(
    train_dataset,           # Training data
    validation_data=val_dataset,  # Validation data (used to monitor performance)
    epochs=5                 # Number of training epochs
)

# Training results from the comments:
# Epoch 1/5: loss: 1.1894 - accuracy: 0.5792 - val_loss: 0.8675 - val_accuracy: 0.7040
# Epoch 2/5: loss: 0.8504 - accuracy: 0.7102 - val_loss: 0.6572 - val_accuracy: 0.7899
# Epoch 3/5: loss: 0.7052 - accuracy: 0.7670 - val_loss: 0.5334 - val_accuracy: 0.8259
# Epoch 4/5: loss: 0.5891 - accuracy: 0.8095 - val_loss: 0.3778 - val_accuracy: 0.8892
# Epoch 5/5: loss: 0.4545 - accuracy: 0.8499 - val_loss: 0.2699 - val_accuracy: 0.9233
# Final validation accuracy: 92.33% - very good performance!

"""
STEP 10: Save the trained model
"""
# Save in Keras format (recommended)
model.save('MCTC_Emotion.keras')

# Also save in H5 format to Google Drive
model.save("/content/drive/MyDrive/MCTC_Emotion5epoch.h5")
print("Model saved successfully!")

"""
STEP 11: Create prediction function
"""
def predict_emotion(text, model, tokenizer):
    """
    This function predicts the emotion of a given text
    
    Parameters:
    - text: Input text string
    - model: Trained emotion classification model
    - tokenizer: BERT tokenizer
    
    Returns:
    - predicted_emotion: The most likely emotion (string)
    - probabilities: Array of probabilities for all 7 emotions
    """
    
    # Preprocess the input text (same as during training)
    inputs = prepare_data(text, tokenizer)
    
    # Get predictions from the model
    # The model outputs probabilities for each of the 7 emotions
    probs = model.predict([inputs['input_ids'], inputs['attention_mask']])[0]
    
    # Find the emotion with the highest probability
    predicted_class_index = np.argmax(probs)
    predicted_emotion = classes[predicted_class_index]
    
    return predicted_emotion, probs

"""
STEP 12: Test the model with example phrases
"""
# Define test phrases with their expected emotions (commented)
test_phrases = [
    "Oh my God!!",  # Expected: surprise
    "I didn't make it!",  # Expected: sad
    "You know how much I love listening to your music, you know, but...",  # Expected: neutral
    "Alright, you did it! Do we have any fruit?",  # Expected: happy
    "i feel insecure and useless",  # Expected: fear
    "I reached into the leper colony and felt a fungal decomposing rat cling to my hair, amid the hum of bloated mosquitoes.",  # Expected: disgust
    "Did it ever occur to you that I might just be that stupid?"  # Expected: anger
]

# Display all emotion classes for reference
print("Emotion classes:", classes)
print("\n" + "="*50)
print("TESTING THE MODEL:")
print("="*50)

# Test each phrase
for phrase in test_phrases:
    # Get prediction for this phrase
    pred_class, probs = predict_emotion(phrase, model, tokenizer)
    
    print(f"\nText: '{phrase}'")
    print(f"Predicted emotion: {pred_class}")
    print(f"Confidence scores:")
    
    # Display probability for each emotion
    for i, emotion in enumerate(classes):
        print(f"  {emotion}: {probs[i]:.4f} ({probs[i]*100:.1f}%)")
    
    print("-" * 40)

"""
STEP 13: Visualize training progress
"""
import matplotlib.pyplot as plt

# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Time")
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss Over Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""
SUMMARY:
This code creates an emotion classification system that:

1. Loads emotion-labeled text data from a zip file
2. Uses BERT tokenizer to convert text into numbers
3. Builds a neural network with BERT as the foundation
4. Trains the model to recognize 7 different emotions
5. Achieves 92.33% accuracy on validation data
6. Can predict emotions for new text with confidence scores

The model works by:
- BERT understands the meaning and context of words
- The dense layer learns emotion-specific patterns
- The output layer produces probability scores for each emotion
- The highest probability determines the predicted emotion

Key concepts:
- Tokenization: Converting text to numbers
- One-hot encoding: Converting labels to binary vectors
- Attention masks: Telling BERT which tokens to focus on
- Softmax: Converting outputs to probabilities
- Validation: Testing on unseen data to prevent overfitting
"""