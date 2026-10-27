# --- Notebook cell 10 ---
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# --- Notebook cell 11 ---
def run_model(name, layer):
    model = Sequential([
        layer(50, activation="tanh",
              input_shape=(LOOK_BACK, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=8,
        verbose=0
    )

    # Predict
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    # Convert to original scale
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)

    actual_train = scaler.inverse_transform(
        y_train.reshape(-1, 1)
    )
    actual_test = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    # Calculate RMSE
    train_rmse = np.sqrt(
        mean_squared_error(actual_train, train_pred)
    )
    test_rmse = np.sqrt(
        mean_squared_error(actual_test, test_pred)
    )

    print(f"\n{name} Results")
    print("Train RMSE:", round(train_rmse, 2))
    print("Test RMSE:", round(test_rmse, 2))

    # Plot loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(name + " Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(name.lower() + "_loss.png")
    plt.close()

    return test_pred, test_rmse, train_rmse

# --- Notebook cell 12 ---
rnn_pred, rnn_rmse, rnn_train_rmse = run_model(
    "RNN", SimpleRNN
)

lstm_pred, lstm_rmse, lstm_train_rmse = run_model(
    "LSTM", LSTM
)

gru_pred, gru_rmse, gru_train_rmse = run_model(
    "GRU", GRU
)

# --- Notebook cell 13 ---
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- Notebook cell 14 ---
plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual", linewidth=2)
plt.plot(rnn_pred, label="RNN")
plt.plot(lstm_pred, label="LSTM")
plt.plot(gru_pred, label="GRU")

# --- Notebook cell 15 ---
plt.title("RNN vs LSTM vs GRU Forecast")
plt.xlabel("Test Time Step")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.tight_layout()
plt.savefig("rnn_vs_lstm_forecast.png")
plt.show()

# --- Notebook cell 16 ---
print("\n===== FINAL RMSE SUMMARY =====")
print("RNN  - Train:", round(rnn_train_rmse, 2),
      "| Test:", round(rnn_rmse, 2))
print("LSTM - Train:", round(lstm_train_rmse, 2),
      "| Test:", round(lstm_rmse, 2))
print("GRU  - Train:", round(gru_train_rmse, 2),
      "| Test:", round(gru_rmse, 2))
