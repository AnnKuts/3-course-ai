import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)

A = fuzz.gaussmf(x, 6, 1.5)

complement = 1 - A

plt.figure(figsize=(10, 5))

plt.plot(x, A, linewidth=2, label='Множина A')
plt.plot(x, complement, 'k--', linewidth=3,
         label='Доповнення NOT A = 1 - A')

plt.title('Доповнення нечіткої множини')
plt.xlabel('x')
plt.ylabel('Ступінь приналежності')
plt.legend()
plt.grid(True)

plt.show()