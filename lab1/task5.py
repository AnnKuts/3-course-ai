import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 500)

z_function = fuzz.zmf(x, 4, 9)
pi_function = fuzz.pimf(x, 3, 6, 12, 16)
s_function = fuzz.smf(x, 11, 17)

plt.figure(figsize=(10, 5))

plt.plot(x, z_function, linewidth=2, label='Z-функція (zmf)')
plt.plot(x, pi_function, linewidth=2, label='PI-функція (pimf)')
plt.plot(x, s_function, linewidth=2, label='S-функція (smf)')

plt.title('Поліноміальні функції приналежності')
plt.xlabel('x')
plt.ylabel('Ступінь приналежності')

plt.legend()
plt.grid(True)
plt.show()

