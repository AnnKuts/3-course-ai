import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_score = np.arange(0, 10.1, 0.1)

average_viewer = fuzz.gbellmf(x_score, 1.5, 4, 5)
critic = fuzz.gbellmf(x_score, 0.8, 8, 3)
optimist = fuzz.gbellmf(x_score, 2.0, 2, 8)

plt.figure(figsize=(10, 5))
plt.plot(x_score, average_viewer, color='blue', linewidth=2, label='Пересічний глядач')
plt.plot(x_score, critic, color='red', linewidth=2, label='Суворий критик')
plt.plot(x_score, optimist, color='green', linewidth=2, label='Оптиміст')

plt.title('Узагальнений дзвін')
plt.xlabel('Бали')
plt.ylabel('Ступінь приналежності')
plt.legend()
plt.grid(True)
plt.show()

