import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, TimeDistributed, Lambda, Input, Bidirectional, BatchNormalization, Dense, ZeroPadding1D
from tensorflow.keras.utils import to_categorical

# Data Paths
train_csv = 'dataset/written_name_train_v2--.csv'
val_csv = 'dataset/written_name_validation_v2--.csv'
test_csv = 'dataset/written_name_test_v2--.csv'
image_train = 'dataset/train_v2/train'
image_val = 'dataset/validation_v2/validation'
image_test = 'dataset/test_v2/test'

# Reading the CSV files
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

# Deleting rows with missing values
train_df.dropna(subset=['IDENTITY'], inplace=True)
val_df.dropna(subset=['IDENTITY'], inplace=True)
test_df.dropna(subset=['IDENTITY'], inplace=True)

# Converting the 'IDENTITY' column to string
train_df['IDENTITY'] = train_df['IDENTITY'].astype(str)
val_df['IDENTITY'] = val_df['IDENTITY'].astype(str)
test_df['IDENTITY'] = test_df['IDENTITY'].astype(str)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 32)):
    try:
        img = Image.open(image_path).convert("L")  # scale to grayscale
        img = img.resize(target_size)  # Change the image size
        img_array = np.array(img) / 255.0  # Normalize the image
        return img_array
    except Exception as e:
        print(f"Nie można otworzyć {image_path}: {e}")
        return None  # Return None if error occurs

# Function to load and process the data
def load_and_process_data(df, image_folder):
    images = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(image_folder, row['FILENAME'])
        img_array = preprocess_image(image_path)

        if img_array is not None:
            images.append(img_array)
            labels.append(row['IDENTITY'])

    return np.array(images), labels

# Load and process the data
X_train, y_train = load_and_process_data(train_df, image_train)
X_val, y_val = load_and_process_data(val_df, image_val)
X_test, y_test = load_and_process_data(test_df, image_test)

# Check the shape of the data
print("Train:", X_train.shape, len(y_train))
print("Validation:", X_val.shape, len(y_val))
print("Test:", X_test.shape, len(y_test))

# Tokenization
tokenizer = Tokenizer(char_level=True)  # Character level tokenization
tokenizer.fit_on_texts(y_train)

# Convert text to sequences
y_train_seq = tokenizer.texts_to_sequences(y_train)
y_val_seq = tokenizer.texts_to_sequences(y_val)
y_test_seq = tokenizer.texts_to_sequences(y_test)

# Padding sequences
max_length = max(max(len(seq) for seq in y_train_seq),
                 max(len(seq) for seq in y_val_seq),
                 max(len(seq) for seq in y_test_seq))

print(f"Max Length: {max_length}") # Print max length

# Zero-padding a sequence to the same length
y_train_padded = pad_sequences(y_train_seq, maxlen=max_length, padding='post')
y_val_padded = pad_sequences(y_val_seq, maxlen=max_length, padding='post')
y_test_padded = pad_sequences(y_test_seq, maxlen=max_length, padding='post')

# Convert padded sequences to categorical
num_classes = len(tokenizer.word_index) + 1  # Add 1 for padding token
y_train_categorical = to_categorical(y_train_padded, num_classes=num_classes)
y_val_categorical = to_categorical(y_val_padded, num_classes=num_classes)
y_test_categorical = to_categorical(y_test_padded, num_classes=num_classes)

# Check the shape of the data
print("Shape X_train:", X_train.shape)
print("Shape y_train_padded:", y_train_padded.shape)
print("Shape y_train_categorical:", y_train_categorical.shape)

# Neural network model - Using a CNN + Bidirectional LSTM architecture
input_img = Input(shape=(32, 128))  # Input shape for images (32, 128)
x = Reshape((32, 128, 1))(input_img)  # Reshape image to match Conv2D input

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Flatten and reshape to match LSTM input
x = Reshape((-1, 64 * 32 // 4))(x)

# Add padding to match target sequence length
x = ZeroPadding1D(padding=(0, 2))(x)  # Padding to extend to 34 steps

# Use Bidirectional LSTM with return_sequences=True
x = Bidirectional(LSTM(128, return_sequences=True))(x)

# Batch normalization
x = BatchNormalization()(x)

# Ensure the output length is exactly 34 (max_length)
x = Lambda(lambda x: x[:, :34, :])(x)  # Crop or pad to 34 if necessary
output = TimeDistributed(Dense(num_classes, activation='softmax'))(x)

# Model setup
model = Model(inputs=input_img, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
model.fit(X_train, y_train_categorical, validation_data=(X_val, y_val_categorical), epochs=10, batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_categorical)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
model.save('model/handwriting_recognition_32_smallDS.keras')
print("Model saved successfully!")