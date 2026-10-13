
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)

duration = np.concatenate([
    rng.normal(55, 25, 100),
    rng.normal(150, 40, 100),
    rng.normal(190, 40, 100)
])

completeness = np.concatenate([
    rng.normal(20, 18, 100),
    rng.normal(80, 17, 100),
    rng.normal(30, 20, 100)
])

duration = np.clip(duration, 10, 300)
completeness = np.clip(completeness, 0, 100)

data = np.column_stack((duration, completeness))
data = data[rng.permutation(len(data))]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    scaled_data.T,
    c=3,
    m=3,
    error=0.005,
    maxiter=100,
    seed=42
)

original_centers = scaler.inverse_transform(centers)
labels = np.argmax(u, axis=0)

for i, center in enumerate(original_centers):
    print(
        f"Кластер {i + 1}: "
        f"тривалість = {center[0]:.1f} с, "
        f"повнота = {center[1]:.1f}%"
    )

plt.figure(figsize=(8, 5))

for i in range(3):
    points = data[labels == i]
    plt.scatter(
        points[:, 0], points[:, 1],
        alpha=0.6,
        label=f"Кластер {i + 1}"
    )

plt.scatter(
    original_centers[:, 0],
    original_centers[:, 1],
    marker="*",
    s=240,
    color="black",
    label="Центри кластерів"
)

plt.xlabel("Тривалість дзвінка (с)")
plt.ylabel("Повнота інформації (%)")
plt.title("Нечітка кластеризація дзвінків AI-агента")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(jm) + 1), jm, marker="o")
plt.xlabel("Номер ітерації")
plt.ylabel("Значення цільової функції")
plt.title("Зміна цільової функції FCM")
plt.grid(True)
plt.show()

print("Приналежності першого дзвінка:", u[:, 0])
print("Кількість ітерацій:", p)
print("FPC:", fpc)
