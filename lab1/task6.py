import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)

A = fuzz.gaussmf(x, 3.5, 1.2)
B = fuzz.gaussmf(x, 6.5, 1.2)

AND = np.fmin(A, B)
OR = np.fmax(A, B)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# AND
axes[0].plot(x, A, linewidth=2, label='Множина A')
axes[0].plot(x, B, linewidth=2, label='Множина B')
axes[0].plot(x, AND, 'k--', linewidth=3,
             label='A AND B = min(A, B)')

axes[0].set_title('Логічне AND')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Ступінь приналежності')
axes[0].legend()
axes[0].grid(True)

# OR
axes[1].plot(x, A, linewidth=2, label='Множина A')
axes[1].plot(x, B, linewidth=2, label='Множина B')
axes[1].plot(x, OR, 'k--', linewidth=3,
             label='A OR B = max(A, B)')

axes[1].set_title('Логічне OR')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Ступінь приналежності')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Мінімаксна інтерпретація логічних операторів')
plt.tight_layout()
plt.show()