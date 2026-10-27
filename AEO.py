
# AUTOENCODER FOR SENTIMENT CLASSIFICATION

# 1. Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, classification_report

# 2. Load dataset
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# 3. Preprocess data
x_train = pad_sequences(
    x_train, maxlen=max_len, padding="post"
)
x_test = pad_sequences(
    x_test, maxlen=max_len, padding="post"
)

x_train = x_train.astype("float32") / vocab_size
x_test = x_test.astype("float32") / vocab_size

# 4. Build Autoencoder
input_layer = Input(shape=(max_len,))

# Encoder
encoded = Dense(128, activation="relu")(input_layer)
encoded = Dense(64, activation="relu")(encoded)
latent = Dense(32, activation="relu", name="latent_layer")(encoded)

# Decoder
decoded = Dense(64, activation="relu")(latent)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dense(max_len, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)

# 5. Compile and train Autoencoder
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# 6. Extract compressed features
encoder = Model(input_layer, latent)

X_train_features = encoder.predict(x_train, verbose=0)
X_test_features = encoder.predict(x_test, verbose=0)

print("Compressed feature shape:", X_train_features.shape)

# 7. Build sentiment classifier
classifier_input = Input(shape=(32,))
x = Dense(16, activation="relu")(classifier_input)
output = Dense(1, activation="sigmoid")(x)

classifier = Model(classifier_input, output)

classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 8. Train classifier
classifier.fit(
    X_train_features, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# 9. Predict sentiment
y_prob = classifier.predict(X_test_features, verbose=0)
y_pred = (y_prob > 0.5).astype(int).ravel()

# 10. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"]
))

# 11. Final result
print("\nResult: Autoencoder-based sentiment classification completed.")
