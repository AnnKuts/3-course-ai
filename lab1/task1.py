import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_temp = np.arange(0, 41, 1)

cold = fuzz.trimf(x_temp, [0, 0, 20])
comfortable = fuzz.trapmf(x_temp, [15, 20, 23, 28])
hot = fuzz.trimf(x_temp, [25, 40, 40])

plt.figure(figsize=(8, 4))
plt.plot(x_temp, cold, 'b', label='Холодно (trimf)')
plt.plot(x_temp, comfortable, 'g', label='Комфортно (trapmf)')
plt.plot(x_temp, hot, 'r', label='Спекотно (trimf)')
plt.title('Температурні функції приналежності')
plt.legend()
plt.grid(True)
plt.show()

