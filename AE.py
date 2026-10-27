import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Documents\Deep Learning\codes-main\IMDB Dataset.csv")
X, y = df.iloc[:, 0].astype(str), df.iloc[:, 1]
y = y.replace({'Negative': 0, 'Positive': 1, 'negative': 0, 'positive': 1}).astype(int)

# Preprocess
vocab_size, max_len = 10000, 200
tok = Tokenizer(num_words=vocab_size)
tok.fit_on_texts(X)
X = pad_sequences(tok.texts_to_sequences(X), maxlen=max_len, padding="post") / vocab_size
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# Autoencoder
i = Input(shape=(max_len,))
x = Dense(128, activation="relu")(i)
x = Dense(64, activation="relu")(x)
z = Dense(32, activation="relu")(x)
x = Dense(64, activation="relu")(z)
x = Dense(128, activation="relu")(x)
o = Dense(max_len, activation="sigmoid")(x)

ae = Model(i, o)
ae.compile(optimizer="adam", loss="mse")
ae.summary()
ae.fit(x_train, x_train, epochs=5, batch_size=128, validation_split=0.2)

# Extract features
enc = Model(i, z)
X_train_features = enc.predict(x_train, verbose=0)
X_test_features = enc.predict(x_test, verbose=0)
print("Compressed feature shape:", X_train_features.shape)

# Classifier
i2 = Input(shape=(32,))
x = Dense(16, activation="relu")(i2)
o2 = Dense(1, activation="sigmoid")(x)  
clf = Model(i2, o2)
clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
clf.fit(X_train_features, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Evaluate
y_pred = (clf.predict(X_test_features, verbose=0) > 0.5).astype(int).ravel()
print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
print("\nResult: Autoencoder-based sentiment classification completed.")