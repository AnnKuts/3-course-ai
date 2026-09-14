import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 500)

one_side = fuzz.sigmf(x, 8, 1.1)
two_side = fuzz.dsigmf(x, 5, 1.4, 14, 1.4)
asymmetric = fuzz.psigmf(x, 3.5, 1.8, 15, -0.55)

plt.figure(figsize=(10, 5))

plt.plot(x, one_side, linewidth=2, label='Одностороння (sigmf)')
plt.plot(x, two_side, linewidth=2, label='Двостороння (dsigmf)')
plt.plot(x, asymmetric, linewidth=2, label='Несиметрична (psigmf)')

plt.title('Сигмоїдні функції приналежності')
plt.xlabel('x')
plt.ylabel('Ступінь приналежності')

plt.legend()
plt.grid(True)
plt.show()


