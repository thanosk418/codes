import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Embedding, LSTM, Concatenate, Dropout

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train[:4000].astype("float32")[..., None] / 255.0
y_train = y_train[:4000]
x_test = x_test[:500].astype("float32")[..., None] / 255.0
y_test = y_test[:500]

names = ["t-shirt", "trouser", "pullover", "dress", "coat",
         "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# 2. Generate captions from class labels
templates = ["a photo of a {}", "this is a {}", "image of a {}"]

def make_caption(label):
    name = names[int(label)]
    caption = random.choice(templates).format(name)
    return "startseq " + caption + " endseq"

train_caps = [make_caption(y) for y in y_train]
test_caps = [make_caption(y) for y in y_test]

print("Example captions:", train_caps[:3])

# 3. Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_caps)

vocab_size = len(tokenizer.word_index) + 1
train_seq = tokenizer.texts_to_sequences(train_caps)
test_seq = tokenizer.texts_to_sequences(test_caps)
max_len = max(len(s) for s in train_seq)

# 4. Prepare next-word training data
X_img, X_seq, Y = [], [], []

for img, seq in zip(x_train, train_seq):
    for i in range(1, len(seq)):
        X_img.append(img)
        X_seq.append(pad_sequences([seq[:i]], maxlen=max_len)[0])
        Y.append(seq[i])

X_img = np.array(X_img)
X_seq = np.array(X_seq)
Y = to_categorical(Y, num_classes=vocab_size)

# 5. CNN encoder
image_input = Input(shape=(28, 28, 1))
a = Conv2D(32, 3, activation="relu", padding="same")(image_input)
a = MaxPooling2D()(a)
a = Conv2D(64, 3, activation="relu", padding="same")(a)
a = MaxPooling2D()(a)
a = GlobalAveragePooling2D()(a)
a = Dense(128, activation="relu")(a)
a = Dropout(0.3)(a)

# 6. LSTM decoder
text_input = Input(shape=(max_len,))
b = Embedding(vocab_size, 128, mask_zero=False)(text_input)
b = Dropout(0.3)(b)
b = LSTM(128)(b)

# 7. Combine CNN and LSTM
combined = Concatenate()([a, b])
combined = Dense(128, activation="relu")(combined)
output = Dense(vocab_size, activation="softmax")(combined)

model = Model([image_input, text_input], output)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# 8. Train model
history = model.fit(
    [X_img, X_seq], Y,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# 9. Plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# 10. Generate caption
index_to_word = {v: k for k, v in tokenizer.word_index.items()}

def generate_caption(image):
    caption = "startseq"

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_len)

        prediction = model.predict(
            [image[None, ...], seq], verbose=0
        )
        word_id = int(np.argmax(prediction))
        word = index_to_word.get(word_id, "")

        if not word or word == "endseq":
            break

        caption += " " + word

    return caption.replace("startseq", "").strip()

# 11. Display sample predictions
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, i in zip(axes.flat, random.sample(range(len(x_test)), 6)):
    caption = generate_caption(x_test[i])
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    ax.set_title("True: " + names[int(y_test[i])] +
                 "\nCaption: " + caption, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Generated caption:", generate_caption(x_test[0]))