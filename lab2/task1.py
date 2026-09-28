import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate


# 1. Початкові функції

x = np.linspace(0, 10, 100)

y = np.sin(x) + np.cos(x / 2)

z = np.sin(2 * np.sqrt(x**2 + y**2)) / (
    np.sqrt(x**2 + y**2) + 0.001
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y, linewidth=2)
plt.title("Функція y(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, z, linewidth=2)
plt.title("Функція z(x, y)")
plt.xlabel("x")
plt.ylabel("z")
plt.grid(True)

plt.tight_layout()
plt.show()


# 2. Створення діапазонів та центрів функцій приналежності

num_x = 6
num_y = 6
num_z = 9

x_universe = np.linspace(min(x), max(x), 500)
y_universe = np.linspace(min(y), max(y), 500)
z_universe = np.linspace(min(z), max(z), 500)

x_centers = np.linspace(min(x), max(x), num_x)
y_centers = np.linspace(min(y), max(y), num_y)
z_centers = np.linspace(min(z), max(z), num_z)

x_step = x_centers[1] - x_centers[0]
y_step = y_centers[1] - y_centers[0]
z_step = z_centers[1] - z_centers[0]


# 3. Функція пошуку найбільшої приналежності

def find_best_function(value, universe, functions):
    values = []

    for function in functions:
        degree = fuzz.interp_membership(
            universe,
            function,
            value
        )

        values.append(degree)

    return np.argmax(values)


# 4. Формування таблиці правил

def create_rules(z_functions):
    rules = {}

    values_table = [
        ["y \\ x"] +
        ["mx" + str(i + 1) for i in range(num_x)]
    ]

    names_table = [
        ["y \\ x"] +
        ["mx" + str(i + 1) for i in range(num_x)]
    ]

    for i in range(num_y):
        values_row = ["my" + str(i + 1)]
        names_row = ["my" + str(i + 1)]

        for j in range(num_x):
            x_value = x_centers[j]
            y_value = y_centers[i]

            z_value = np.sin(
                2 * np.sqrt(x_value**2 + y_value**2)
            ) / (
                np.sqrt(x_value**2 + y_value**2) + 0.001
            )

            best_z = find_best_function(
                z_value,
                z_universe,
                z_functions
            )

            rules[(j, i)] = best_z

            values_row.append(round(z_value, 3))
            names_row.append("mf" + str(best_z + 1))

        values_table.append(values_row)
        names_table.append(names_row)

    return rules, values_table, names_table


# 5. Моделювання функції

def calculate_model(
    x_functions,
    y_functions,
    rules,
    diagonal=False
):
    z_model = []

    for x_value, y_value in zip(x, y):

        x_membership = []
        y_membership = []

        for function in x_functions:
            value = fuzz.interp_membership(
                x_universe,
                function,
                x_value
            )

            x_membership.append(value)

        for function in y_functions:
            value = fuzz.interp_membership(
                y_universe,
                function,
                y_value
            )

            y_membership.append(value)

        result = 0
        total_weight = 0

        for i in range(num_y):
            for j in range(num_x):

                if diagonal and i != j:
                    continue

                weight = min(
                    x_membership[j],
                    y_membership[i]
                )

                if weight > 0:
                    z_index = rules[(j, i)]

                    result += (
                        weight *
                        z_centers[z_index]
                    )

                    total_weight += weight

        if total_weight > 0:
            z_model.append(
                result / total_weight
            )
        else:
            z_model.append(
                np.mean(z_centers)
            )

    return np.array(z_model)


# 6. Розрахунок похибок

def calculate_errors(real, model):
    mse = mean_squared_error(real, model)
    mae = mean_absolute_error(real, model)

    relative_error = np.mean(
        np.abs(
            (real + 1) -
            (model + 1)
        ) /
        np.abs(real + 1)
    ) * 100

    return mse, mae, relative_error


# 7. Трикутні функції приналежності

x_trimf = []
y_trimf = []
z_trimf = []

for center in x_centers:
    function = fuzz.trimf(
        x_universe,
        [
            center - x_step,
            center,
            center + x_step
        ]
    )

    x_trimf.append(function)

for center in y_centers:
    function = fuzz.trimf(
        y_universe,
        [
            center - y_step,
            center,
            center + y_step
        ]
    )

    y_trimf.append(function)

for center in z_centers:
    function = fuzz.trimf(
        z_universe,
        [
            center - z_step,
            center,
            center + z_step
        ]
    )

    z_trimf.append(function)


plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
for function in x_trimf:
    plt.plot(
        x_universe,
        function,
        linewidth=2
    )
plt.title("Трикутні функції приналежності X")
plt.xlabel("x")
plt.ylabel("Ступінь приналежності")
plt.grid(True)

plt.subplot(3, 1, 2)
for function in y_trimf:
    plt.plot(
        y_universe,
        function,
        linewidth=2
    )
plt.title("Трикутні функції приналежності Y")
plt.xlabel("y")
plt.ylabel("Ступінь приналежності")
plt.grid(True)

plt.subplot(3, 1, 3)
for function in z_trimf:
    plt.plot(
        z_universe,
        function,
        linewidth=2
    )
plt.title("Трикутні функції приналежності Z")
plt.xlabel("z")
plt.ylabel("Ступінь приналежності")
plt.grid(True)

plt.tight_layout()
plt.show()


rules_trimf, values_table, names_table = create_rules(
    z_trimf
)

print("Таблиця значень")
print(
    tabulate(
        values_table,
        headers="firstrow",
        tablefmt="grid"
    )
)

print("\nТаблиця правил")
print(
    tabulate(
        names_table,
        headers="firstrow",
        tablefmt="grid"
    )
)


z_trimf_model = calculate_model(
    x_trimf,
    y_trimf,
    rules_trimf
)

mse_trimf, mae_trimf, relative_trimf = calculate_errors(
    z,
    z_trimf_model
)

print("\nТрикутні функції")
print("MSE:", mse_trimf)
print("MAE:", mae_trimf)
print(
    "Середня відносна похибка:",
    round(relative_trimf, 2),
    "%"
)

plt.figure(figsize=(10, 5))
plt.plot(
    x,
    z,
    linewidth=2,
    label="Точна функція"
)
plt.plot(
    x,
    z_trimf_model,
    linewidth=2,
    label="Нечітка модель"
)
plt.title("Модель з трикутними функціями")
plt.xlabel("x")
plt.ylabel("z")
plt.legend()
plt.grid(True)
plt.show()


# 8. Трапецієподібні функції приналежності

x_trapmf = []
y_trapmf = []
z_trapmf = []

for center in x_centers:
    function = fuzz.trapmf(
        x_universe,
        [
            center - x_step,
            center - x_step / 3,
            center + x_step / 3,
            center + x_step
        ]
    )

    x_trapmf.append(function)

for center in y_centers:
    function = fuzz.trapmf(
        y_universe,
        [
            center - y_step,
            center - y_step / 3,
            center + y_step / 3,
            center + y_step
        ]
    )

    y_trapmf.append(function)

for center in z_centers:
    function = fuzz.trapmf(
        z_universe,
        [
            center - z_step,
            center - z_step / 3,
            center + z_step / 3,
            center + z_step
        ]
    )

    z_trapmf.append(function)


plt.figure(figsize=(10, 5))

for function in x_trapmf:
    plt.plot(
        x_universe,
        function,
        linewidth=2
    )

plt.title("Трапецієподібні функції приналежності X")
plt.xlabel("x")
plt.ylabel("Ступінь приналежності")
plt.grid(True)
plt.show()


rules_trapmf, values_table, names_table = create_rules(
    z_trapmf
)

z_trapmf_model = calculate_model(
    x_trapmf,
    y_trapmf,
    rules_trapmf
)

mse_trapmf, mae_trapmf, relative_trapmf = calculate_errors(
    z,
    z_trapmf_model
)

print("\nТрапецієподібні функції")
print("MSE:", mse_trapmf)
print("MAE:", mae_trapmf)
print(
    "Середня відносна похибка:",
    round(relative_trapmf, 2),
    "%"
)

plt.figure(figsize=(10, 5))
plt.plot(
    x,
    z,
    linewidth=2,
    label="Точна функція"
)
plt.plot(
    x,
    z_trapmf_model,
    linewidth=2,
    label="Нечітка модель"
)
plt.title("Модель з трапецієподібними функціями")
plt.xlabel("x")
plt.ylabel("z")
plt.legend()
plt.grid(True)
plt.show()


# 9. Гаусові функції приналежності

x_gaussmf = []
y_gaussmf = []
z_gaussmf = []

for center in x_centers:
    function = fuzz.gaussmf(
        x_universe,
        center,
        x_step / 2
    )

    x_gaussmf.append(function)

for center in y_centers:
    function = fuzz.gaussmf(
        y_universe,
        center,
        y_step / 2
    )

    y_gaussmf.append(function)

for center in z_centers:
    function = fuzz.gaussmf(
        z_universe,
        center,
        z_step / 2
    )

    z_gaussmf.append(function)


plt.figure(figsize=(10, 5))

for function in x_gaussmf:
    plt.plot(
        x_universe,
        function,
        linewidth=2
    )

plt.title("Гаусові функції приналежності X")
plt.xlabel("x")
plt.ylabel("Ступінь приналежності")
plt.grid(True)
plt.show()


rules_gaussmf, values_table, names_table = create_rules(
    z_gaussmf
)

z_gaussmf_model = calculate_model(
    x_gaussmf,
    y_gaussmf,
    rules_gaussmf
)

mse_gaussmf, mae_gaussmf, relative_gaussmf = calculate_errors(
    z,
    z_gaussmf_model
)

print("\nГаусові функції")
print("MSE:", mse_gaussmf)
print("MAE:", mae_gaussmf)
print(
    "Середня відносна похибка:",
    round(relative_gaussmf, 2),
    "%"
)

plt.figure(figsize=(10, 5))
plt.plot(
    x,
    z,
    linewidth=2,
    label="Точна функція"
)
plt.plot(
    x,
    z_gaussmf_model,
    linewidth=2,
    label="Нечітка модель"
)
plt.title("Модель з гаусовими функціями")
plt.xlabel("x")
plt.ylabel("z")
plt.legend()
plt.grid(True)
plt.show()


# 10. Порівняння форм функцій приналежності

comparison = [
    [
        "Тип функції",
        "MSE",
        "MAE",
        "Відносна похибка, %"
    ],
    [
        "Трикутна",
        round(mse_trimf, 5),
        round(mae_trimf, 5),
        round(relative_trimf, 2)
    ],
    [
        "Трапецієподібна",
        round(mse_trapmf, 5),
        round(mae_trapmf, 5),
        round(relative_trapmf, 2)
    ],
    [
        "Гаусова",
        round(mse_gaussmf, 5),
        round(mae_gaussmf, 5),
        round(relative_gaussmf, 2)
    ]
]

print("\nПорівняння функцій приналежності")
print(
    tabulate(
        comparison,
        headers="firstrow",
        tablefmt="grid"
    )
)


# 11. Повна та діагональна база правил

z_full = calculate_model(
    x_trimf,
    y_trimf,
    rules_trimf
)

z_diagonal = calculate_model(
    x_trimf,
    y_trimf,
    rules_trimf,
    diagonal=True
)

mse_full, mae_full, relative_full = calculate_errors(
    z,
    z_full
)

mse_diagonal, mae_diagonal, relative_diagonal = calculate_errors(
    z,
    z_diagonal
)

rules_comparison = [
    [
        "Модель",
        "Кількість правил",
        "MSE",
        "MAE",
        "Відносна похибка, %"
    ],
    [
        "Повна",
        36,
        round(mse_full, 5),
        round(mae_full, 5),
        round(relative_full, 2)
    ],
    [
        "Діагональна",
        6,
        round(mse_diagonal, 5),
        round(mae_diagonal, 5),
        round(relative_diagonal, 2)
    ]
]

print("\nПорівняння повної та діагональної баз")
print(
    tabulate(
        rules_comparison,
        headers="firstrow",
        tablefmt="grid"
    )
)

plt.figure(figsize=(10, 5))

plt.plot(
    x,
    z,
    linewidth=2,
    label="Точна функція"
)

plt.plot(
    x,
    z_full,
    linewidth=2,
    label="Повна база правил"
)

plt.plot(
    x,
    z_diagonal,
    linewidth=2,
    label="Діагональна база правил"
)

plt.title("Порівняння повної та діагональної баз правил")
plt.xlabel("x")
plt.ylabel("z")
plt.legend()
plt.grid(True)
plt.show()