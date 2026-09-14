import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)

A = fuzz.gaussmf(x, 3.5, 1.2)
B = fuzz.gaussmf(x, 6.5, 1.2)

conjunction = A * B
disjunction = A + B - conjunction

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Кон'юнкція
axes[0].plot(x, A, linewidth=2, label='Множина A')
axes[0].plot(x, B, linewidth=2, label='Множина B')
axes[0].plot(x, conjunction, 'k--', linewidth=3,
             label='Кон’юнкція: A · B')

axes[0].set_title('Кон’юнкція (AND)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Ступінь приналежності')
axes[0].legend()
axes[0].grid(True)

# Диз'юнкція
axes[1].plot(x, A, linewidth=2, label='Множина A')
axes[1].plot(x, B, linewidth=2, label='Множина B')
axes[1].plot(x, disjunction, 'k--', linewidth=3,
             label='Диз’юнкція: A + B - A · B')

axes[1].set_title('Диз’юнкція (OR)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Ступінь приналежності')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Вірогідна інтерпретація логічних операторів')
plt.tight_layout()
plt.show()