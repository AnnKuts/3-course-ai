from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEED = 1
POP_SIZE = 10
GENERATIONS = 100
MUTATION_RATE = 0.1
RUN_SEEDS = [1, 2, 3, 4, 5]

folder = Path(__file__).resolve().parent


def f1(x):
    return np.sin(x) + np.cos(x / 2)


def f2(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    return np.sin(2 * r) / (r + 0.001)


def mean_distance(population):
    diff = population[:, None, :] - population[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    n = len(population)
    return dist.sum() / (n * (n - 1))


def tournament(population, scores, rng):
    i, j = rng.integers(0, len(population), 2)
    return population[i] if scores[i] > scores[j] else population[j]


def genetic_algorithm(func, bounds, maximize, seed):
    rng = np.random.default_rng(seed)
    low, high = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)

    population = rng.uniform(low, high, (POP_SIZE, dim))
    history = {"best": [], "mean": [], "distance": []}
    best_point, best_score = None, -np.inf

    for generation in range(GENERATIONS):
        values = func(*population.T)
        scores = values if maximize else -values

        best = np.argmax(scores)
        if scores[best] > best_score:
            best_point, best_score = population[best].copy(), scores[best]

        history["best"].append(best_score if maximize else -best_score)
        history["mean"].append(values.mean())
        history["distance"].append(mean_distance(population))

        children = [population[best].copy()]
        while len(children) < POP_SIZE:
            parent1 = tournament(population, scores, rng)
            parent2 = tournament(population, scores, rng)

            alpha = rng.random()
            child = alpha * parent1 + (1 - alpha) * parent2

            mask = rng.random(dim) < MUTATION_RATE
            child[mask] += rng.normal(0, 0.1 * (high - low), dim)[mask]
            children.append(np.clip(child, low, high))

        population = np.array(children)

    return best_point, history["best"][-1], history


def plot_progress(axes, history, best_point, names, title):
    generations = np.arange(1, GENERATIONS + 1)
    ax = axes[0]
    ax.plot(generations, history["best"], label="Найкраще значення")
    ax.plot(generations, history["mean"], "--", label="Середнє значення")
    ax.set_title(f"{title}\nнайкраще = {history['best'][-1]:.5f}")
    ax.set_xlabel("Покоління")
    ax.set_ylabel("Значення функції")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    bars = ax.bar(names, best_point, color="tab:green", width=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlim(-0.8, len(names) - 0.2)
    ax.margins(y=0.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Найкраща особина")
    ax.set_xlabel("Змінна")
    ax.set_ylabel("Значення змінної")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.plot(generations, history["distance"], color="tab:purple")
    ax.set_title("Середня відстань між особинами")
    ax.set_xlabel("Покоління")
    ax.set_ylabel("Відстань")
    ax.grid(alpha=0.3)


def print_history(history):
    best = np.array(history["best"])
    reached = np.argmax(np.abs(best - best[-1]) < 1e-4) + 1
    print(f"  Найкраще значення: 1-ше покоління {best[0]:.5f}, "
          f"досягнуто {best[-1]:.5f} у поколінні {reached}")
    print(f"  Середнє значення: 1-ше покоління {history['mean'][0]:.5f}, "
          f"останнє {history['mean'][-1]:.5f}")
    tail = np.mean(history["distance"][50:])
    print(f"  Середня відстань: 1-ше покоління {history['distance'][0]:.4f}, "
          f"середня за останні 50 поколінь {tail:.4f}")


def print_runs(func, bounds, maximize, names):
    print("Незалежні запуски:")
    for seed in RUN_SEEDS:
        point, value, _ = genetic_algorithm(func, bounds, maximize, seed)
        coords = ", ".join(f"{n} = {v:.4f}" for n, v in zip(names, point))
        print(f"  seed = {seed}: {coords}, значення = {value:.5f}")


bounds1 = np.array([[-4.0, 1.0]])
x_best, y_best, history1 = genetic_algorithm(f1, bounds1, maximize=False,
                                             seed=SEED)
print("Завдання 1. Мінімізація y(x) = sin(x) + cos(x/2)")
print(f"  ГА: x = {x_best[0]:.4f}, y(x) = {y_best:.5f}")
print_history(history1)

x_grid = np.linspace(-4, 1, 100001)
y_grid = f1(x_grid)
k = np.argmin(y_grid)
print(f"  Перевірка на сітці: x = {x_grid[k]:.4f}, y(x) = {y_grid[k]:.5f}")
print_runs(f1, bounds1, False, ["x"])

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
axes = axes.ravel()
plot_progress(axes, history1, x_best, ["x"], "Мінімізація y(x)")
ax = axes[3]
ax.plot(x_grid, y_grid, label="y(x) = sin(x) + cos(x/2)")
ax.plot(x_best, y_best, "ro", markersize=8,
        label=f"Мінімум ГА ({x_best[0]:.3f}; {y_best:.3f})")
ax.set_title("Графік функції")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(folder / "task1_minimum.png", dpi=150)
plt.close(fig)

bounds2 = np.array([[-1.0, 3.0], [-1.0, 3.0]])
xy_best, z_best, history2 = genetic_algorithm(f2, bounds2, maximize=True,
                                              seed=SEED)
print("\nЗавдання 2. Максимізація z(x, y) = sin(2r) / (r + 0.001)")
print(f"  ГА: x = {xy_best[0]:.4f}, y = {xy_best[1]:.4f}, "
      f"r = {np.hypot(*xy_best):.4f}, z = {z_best:.5f}")
print_history(history2)

grid = np.linspace(-1, 3, 4001)
X, Y = np.meshgrid(grid, grid)
Z = f2(X, Y)
i, j = np.unravel_index(np.argmax(Z), Z.shape)
print(f"  Перевірка на сітці: x = {X[i, j]:.4f}, y = {Y[i, j]:.4f}, "
      f"r = {np.hypot(X[i, j], Y[i, j]):.4f}, z = {Z[i, j]:.5f}")
print_runs(f2, bounds2, True, ["x", "y"])

fig = plt.figure(figsize=(12, 8.5))
axes = [fig.add_subplot(2, 2, n) for n in (1, 2, 3)]
plot_progress(axes, history2, xy_best, ["x", "y"], "Максимізація z(x, y)")
ax = fig.add_subplot(2, 2, 4, projection="3d")
Xs, Ys = np.meshgrid(np.linspace(-1, 3, 201), np.linspace(-1, 3, 201))
ax.plot_surface(Xs, Ys, f2(Xs, Ys), cmap="viridis", alpha=0.8, linewidth=0)
ax.scatter(*xy_best, z_best, color="red", s=40, depthshade=False)
ax.set_title(f"Поверхня функції, максимум ГА z = {z_best:.3f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.tight_layout()
fig.savefig(folder / "task2_maximum.png", dpi=150)
plt.close(fig)
