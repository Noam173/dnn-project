import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import Data_Manipulation as dp
from sklearn.preprocessing import StandardScaler


def create_model(batch_size: int, num_epoch: int) -> dict:
    tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

    x_train, y_train = dp.Get_train()
    x_val, y_val = dp.Get_val()
    x_test, y_test = dp.Get_test()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.values)
    x_val = scaler.transform(x_val.values)
    x_test = scaler.transform(x_test.values)

    x_train, y_train = tensor(x_train), tensor(y_train)
    x_val, y_val = tensor(x_val), tensor(y_val)
    x_test, y_test = tensor(x_test), tensor(y_test)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
    )

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))

    model.add(Dense(128, activation="relu"))

    model.add(Dense(64, activation="relu"))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(32, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    hist = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epoch,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
    )

    print("\n\nModel summary:")
    print(model.summary())

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}, {loss:.2f}")

    return hist.history
