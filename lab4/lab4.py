import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

x = np.linspace(-5, 5, 1000)
y = np.sin(x) + np.cos(x / 2)

r = np.sqrt(x**2 + y**2)
z = np.sin(2 * r) / (r + 0.001)

X = np.column_stack((x, y))


# Feedforward Backpropagation
def feedforward(neurons):
    inputs = keras.Input(shape=(2,))
    h = layers.Dense(neurons, activation="relu")(inputs)
    outputs = layers.Dense(1)(h)
    return keras.Model(inputs, outputs)


# Cascade-forward Backpropagation
def cascade(sizes):
    inputs = keras.Input(shape=(2,))
    previous = [inputs]

    for neurons in sizes:
        layer_input = (
            previous[0] if len(previous) == 1
            else layers.Concatenate()(previous)
        )
        h = layers.Dense(neurons, activation="relu")(layer_input)
        previous.append(h)

    outputs = layers.Dense(1)(
        layers.Concatenate()(previous)
    )

    return keras.Model(inputs, outputs)


# Elman Backpropagation
def elman(sizes):
    inputs = keras.Input(shape=(2,))
    h = layers.Reshape((2, 1))(inputs)

    for i, neurons in enumerate(sizes):
        h = layers.SimpleRNN(
            neurons,
            activation="tanh",
            return_sequences=(i < len(sizes) - 1)
        )(h)

    outputs = layers.Dense(1)(h)
    return keras.Model(inputs, outputs)


models = {
    "Feedforward 1x10": feedforward(10),
    "Feedforward 1x20": feedforward(20),
    "Cascade 1x20": cascade([20]),
    "Cascade 2x10": cascade([10, 10]),
    "Elman 1x15": elman([15]),
    "Elman 3x5": elman([5, 5, 5])
}

histories = {}
predictions = {}
results = []

for name, model in models.items():
    print("Навчання:", name)

    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X, z,
        epochs=200,
        batch_size=100,
        verbose=0
    )

    predicted = model.predict(X, verbose=0).ravel()

    histories[name] = history.history["loss"]
    predictions[name] = predicted

    mse = np.mean((z - predicted) ** 2)

    relative_error = np.mean(
        np.abs(z - predicted) / np.abs(z + 1)
    ) * 100

    results.append((name, mse, relative_error))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, (name, loss) in zip(
    axes.flat, histories.items()
):
    ax.plot(range(1, len(loss) + 1), loss)
    ax.set_title(name)
    ax.set_xlabel("Епоха")
    ax.set_ylabel("MSE")
    ax.grid(True)

fig.suptitle("Помилка під час навчання шести нейромереж")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, (name, predicted) in zip(
    axes.flat, predictions.items()
):
    ax.plot(x, z, label="Справжня функція")
    ax.plot(x, predicted, label="Прогноз")

    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend()
    ax.grid(True)

fig.suptitle("Порівняння прогнозів із функцією варіанта №1")
fig.tight_layout()
plt.show()

print("\n{:<21} {:>12} {:>23}".format(
    "Модель", "MSE", "Відносна похибка, %"
))
print("-" * 59)

for name, mse, relative_error in sorted(
    results, key=lambda item: item[2]
):
    print(
        f"{name:<21} {mse:>12.6f} "
        f"{relative_error:>23.3f}"
    )
