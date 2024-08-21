import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tkinter as tk
from tkinter import simpledialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("movies.csv")

# Handling NaNs and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Define numerical and categorical columns
numerical_cols = ['Year', 'Runtime', 'No.of.Ratings']
categorical_cols = ['Certificate', 'Overview', 'Movie']

# Fill missing values in numerical columns with mean
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Fill missing values in categorical columns
df['Certificate'].fillna('Unknown', inplace=True)
df['Overview'].fillna('', inplace=True)

# Encode 'Certificate' feature
certificate_encoder = OneHotEncoder()
certificate_encoded = certificate_encoder.fit_transform(df[['Certificate']]).toarray()

# Combine numerical and encoded features
X_numerical = df[numerical_cols].values
X_combined = np.concatenate([X_numerical, certificate_encoded], axis=1)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_combined)

# Simulate interaction data (user, movie, timestamp)
interactions = pd.DataFrame({
    'user_id': np.random.randint(1, 100, 1000),
    'movie_id': np.random.choice(df.index, 1000),
    'timestamp': pd.date_range(start='2020-01-01', periods=1000, freq='H')
})

# Merge interaction data with movie data
interactions = interactions.merge(df, left_on='movie_id', right_index=True)
interactions = interactions.merge(pd.DataFrame(X_scaled, columns=[f'feature_{i}' for i in range(X_scaled.shape[1])]), left_on='movie_id', right_index=True)
interactions = interactions.sort_values(by=['user_id', 'timestamp'])

# Function to create sequences
def create_sequences(interactions, max_sequence_length):
    sequences = []
    for user_id, group in interactions.groupby('user_id'):
        user_sequences = group.iloc[:, -X_scaled.shape[1]:].values
        for i in range(1, len(user_sequences)):
            sequences.append(user_sequences[max(0, i-max_sequence_length):i])
    return sequences

max_sequence_length = 20  # Set desired sequence length
sequences = create_sequences(interactions, max_sequence_length)

X = np.zeros((len(sequences), max_sequence_length, X_scaled.shape[1]), dtype=np.float32)
y = np.zeros((len(sequences), max_sequence_length), dtype=np.int32)

for i, seq in enumerate(sequences):
    seq_length = min(len(seq), max_sequence_length)
    for t in range(seq_length):
        X[i, t, :] = seq[t]
        if t < seq_length - 1:
            y[i, t] = seq[t+1, 1]
    if seq_length < max_sequence_length:
        X[i, seq_length:, :] = seq[-1]

X = X[:, :max_sequence_length, :]
y = y[:, :max_sequence_length]

# Convert y to one-hot encoding
num_classes = len(df)
y_train_onehot = to_categorical(y, num_classes=num_classes).astype(np.float32)

# Define and compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

model = Sequential([
    GRU(units=128, input_shape=(max_sequence_length, X_scaled.shape[1]), return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
    Dropout(0.5),  # Increase dropout for more regularization
    GRU(units=128, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Split data and train the model
# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y_train_onehot, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=20, batch_size=200, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Improved function to recommend movies with added randomness
def recommend_movies(user_id, entered_movie_name, top_n=10):
    user_sequence = interactions[interactions['user_id'] == user_id].iloc[:, -X_scaled.shape[1]:].values
    user_sequence_padded = np.zeros((1, max_sequence_length, X_scaled.shape[1]))
    user_sequence_length = min(max_sequence_length - 1, len(user_sequence))
    user_sequence_padded[0, 1:user_sequence_length+1, :] = user_sequence[-user_sequence_length:]
    
    entered_movie_features = extract_features_by_name(entered_movie_name)
    user_sequence_padded[0, 0, :] = entered_movie_features
    
    user_preferences = model.predict(user_sequence_padded)[0][-1]
    
    # Adding randomness to the top-n recommendations
    top_indices = np.argsort(user_preferences)[::-1][:top_n * 2]
    random.shuffle(top_indices)
    recommended_movie_names = df.iloc[top_indices[:top_n]]['Movie'].values
    
    return recommended_movie_names

# Function to extract features by movie name
def extract_features_by_name(movie_name):
    movie_row = df[df['Movie'] == movie_name]
    if movie_row.empty:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")
    entered_movie_features = movie_row[numerical_cols].values
    certificate_encoded = certificate_encoder.transform(movie_row[['Certificate']]).toarray()
    entered_movie_features_combined = np.concatenate([entered_movie_features, certificate_encoded], axis=1)
    entered_movie_features_scaled = scaler.transform(entered_movie_features_combined)
    return entered_movie_features_scaled.reshape(1, 1, -1)

# GUI for input and output
def get_movie_name():
    while True:
        movie_name = simpledialog.askstring("Movie Recommendation", "Enter the movie name (or type 'exit' to quit):")
        if not movie_name or movie_name.lower() == 'exit':
            break  # Exit the loop if no input or 'exit' is provided
        user_id = random.randint(1, 1000)
        try:
            recommended_movies = recommend_movies(user_id, movie_name, top_n=10)
            result_text = f"Movie Entered: {movie_name}\nRecommended Movies:\n" + "\n".join(recommended_movies)
            messagebox.showinfo("Recommended Movies", result_text)
        except ValueError as e:
            messagebox.showerror("Error", str(e))

# Define user_id
user_id = random.randint(1, 1000)

# Run GUI
root = tk.Tk()
root.withdraw()  # Hide the root window
get_movie_name()
root.destroy()  # Close the root window

# Distribution of movie certificates
plt.figure(figsize=(10, 6))
df['Certificate'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Distribution of Movie Certificates')
plt.ylabel('')
plt.show()

# Visualizing the distribution of movie years
plt.figure(figsize=(10, 6))
sns.histplot(df['Year'], bins=30, kde=True)
plt.title('Distribution of Movie Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.show()

# Visualizing runtime distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Runtime'], bins=30, kde=True)
plt.title('Distribution of Movie Runtimes')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Frequency')
plt.show()
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Plot training and validation loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Predict on the test set
y_pred = model.predict(X_test)

# Convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=-1)
y_true_classes = np.argmax(y_test, axis=-1)

# Flatten the arrays for evaluation
y_pred_classes_flat = y_pred_classes.flatten()
y_true_classes_flat = y_true_classes.flatten()

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes_flat, y_pred_classes_flat)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate precision, recall, and F1-score
precision = precision_score(y_true_classes_flat, y_pred_classes_flat, average='weighted')
recall = recall_score(y_true_classes_flat, y_pred_classes_flat, average='weighted')
f1 = f1_score(y_true_classes_flat, y_pred_classes_flat, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
  

# Function to visualize recommendations
def visualize_recommendations(user_id, entered_movie_name, top_n=10):
    try:
        recommended_movies = recommend_movies(user_id, entered_movie_name, top_n)
        recommended_df = df[df['Movie'].isin(recommended_movies)]
        plt.figure(figsize=(12, 8))
        sns.barplot(y=recommended_df['Movie'], x=recommended_df['No.of.Ratings'], palette='viridis')
        plt.title('Top Recommended Movies')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Movies')
        plt.show()
    except ValueError as e:
        print(str(e))

# Visualize recommendations for a random user and a specific movie
user_id = random.randint(1, 1000)
entered_movie_name = 'Inception'  # Example movie name
visualize_recommendations(user_id, entered_movie_name, top_n=10)