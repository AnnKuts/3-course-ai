import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_bp = np.arange(80, 181, 1)

low_bp = fuzz.gaussmf(x_bp, 90, 8)
high_bp = fuzz.gaussmf(x_bp, 150, 15)
normal_bp = fuzz.gauss2mf(x_bp, 115, 6, 125, 12)

plt.figure(figsize=(10, 5))
plt.plot(x_bp, low_bp, color='blue', linewidth=2, label='Низький (gaussmf)')
plt.plot(x_bp, normal_bp, color='green', linewidth=2, label='Нормальний (gauss2mf)')
plt.plot(x_bp, high_bp, color='red', linewidth=2, label='Високий (gaussmf)')

plt.title('Гаусові функції приналежності')
plt.xlabel('Систолічний тиск (мм рт. ст.)')
plt.ylabel('Ступінь приналежності')
plt.legend()
plt.grid(True)
plt.show()

