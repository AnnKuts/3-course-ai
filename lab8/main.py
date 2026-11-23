from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SEED = 1
folder = Path(__file__).resolve().parent

A, B = 80, 60

SIZES = np.array([
    [24, 20],
    [16, 12],
    [12, 10],
    [10, 8],
    [20, 10],
    [8, 6],
    [14, 14],
    [18, 8],
    [10, 10],
    [22, 12],
    [6, 12],
    [12, 16],
])
N = len(SIZES)

CONNECTIONS = [
    (1, 2, 3), (1, 3, 1), (1, 4, 2), (1, 7, 2), (1, 9, 2), (1, 11, 1),
    (2, 6, 1), (3, 5, 2), (3, 6, 1), (3, 12, 2), (5, 12, 1), (7, 8, 3),
    (7, 6, 1), (9, 10, 2), (9, 11, 1), (10, 5, 1), (4, 6, 1), (11, 12, 1),
]
C = np.zeros((N, N))
for i, j, c in CONNECTIONS:
    C[i - 1, j - 1] = C[j - 1, i - 1] = c

POP_SIZE = 60
MAX_GENERATIONS = 1000
STALL_GENERATIONS = 100
TOLERANCE = 0.1
ELITE = 2
MUTATION_PROB = 0.5
MUTATION_STEP = 0.15
K = 0.05

L_MAX = N ** 2 * np.sqrt(A ** 2 + B ** 2)


def wire_length(z):
    diff = z[:, None, :] - z[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    return (C * d).sum() / 2


def overlap_area(z):
    dx = 0.5 * (SIZES[:, None, 0] + SIZES[None, :, 0]) - np.abs(z[:, None, 0] - z[None, :, 0])
    dy = 0.5 * (SIZES[:, None, 1] + SIZES[None, :, 1]) - np.abs(z[:, None, 1] - z[None, :, 1])
    dx = np.minimum(dx, np.minimum(SIZES[:, None, 0], SIZES[None, :, 0]))
    dy = np.minimum(dy, np.minimum(SIZES[:, None, 1], SIZES[None, :, 1]))
    s = np.clip(dx, 0, None) * np.clip(dy, 0, None)
    return np.triu(s, 1).sum()


def objective(z):
    return K * wire_length(z) / L_MAX + overlap_area(z) / (N * A * B)


def random_layout(rng):
    low, high = SIZES / 2, np.array([A, B]) - SIZES / 2
    return rng.uniform(low, high)


def clip_to_field(z):
    return np.clip(z, SIZES / 2, np.array([A, B]) - SIZES / 2)


def selection_probs(values):
    fitness = 1 / values
    ps = fitness / fitness.sum()
    if ps.max() == ps.min():
        return np.full(len(ps), 1 / len(ps))
    psl = (ps - ps.min()) / (ps.max() - ps.min())
    return psl / psl.sum()


def crossover(parent1, parent2, rng):
    point = rng.integers(1, N)
    return np.vstack([parent1[:point], parent2[point:]])


def mutate(z, mutation_prob, rng):
    z = z.copy()
    if rng.random() < mutation_prob:
        i = rng.integers(N)
        z[i] += rng.normal(0, MUTATION_STEP, 2) * np.array([A, B])
    return clip_to_field(z)


def genetic_algorithm(mutation_prob=MUTATION_PROB, seed=SEED):
    rng = np.random.default_rng(seed)
    population = np.array([random_layout(rng) for _ in range(POP_SIZE)])
    history = {"best": [], "feasible": []}
    initial = None
    best, best_length, last_improvement = None, np.inf, 0

    for generation in range(MAX_GENERATIONS):
        values = np.array([objective(z) for z in population])
        order = np.argsort(values)
        population, values = population[order], values[order]
        history["best"].append(values[0])
        if initial is None:
            initial = population[0].copy()

        for z in population:
            if overlap_area(z) == 0 and wire_length(z) < best_length - TOLERANCE:
                best, best_length = z.copy(), wire_length(z)
                last_improvement = generation
        history["feasible"].append(objective(best) if best is not None else np.nan)

        if best is not None and generation - last_improvement >= STALL_GENERATIONS:
            break

        probs = selection_probs(values)
        children = [population[i].copy() for i in range(ELITE)]
        while len(children) < POP_SIZE:
            i, j = rng.choice(POP_SIZE, 2, p=probs)
            child = crossover(population[i], population[j], rng)
            children.append(mutate(child, mutation_prob, rng))
        population = np.array(children)

    return initial, best, history


def print_layout(title, z):
    print(title)
    print(f"  Довжина з'єднань L = {wire_length(z):.2f} мм")
    print(f"  Площа перекриття S = {overlap_area(z):.2f} мм^2")
    print(f"  Цільова функція F = {objective(z):.6f}")


def draw_layout(ax, z, title):
    ax.add_patch(Rectangle((0, 0), A, B, fill=False, lw=2))
    for i, j, c in CONNECTIONS:
        p, q = z[i - 1], z[j - 1]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="gray", lw=0.6 * c, alpha=0.6)
    for k, ((x, y), (w, h)) in enumerate(zip(z, SIZES)):
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, alpha=0.5,
                               facecolor=f"C{k % 10}", edgecolor="black"))
        ax.text(x, y, str(k + 1), ha="center", va="center", fontweight="bold")
    ax.set_xlim(-2, A + 2)
    ax.set_ylim(-2, B + 2)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nL = {wire_length(z):.1f} мм, S = {overlap_area(z):.1f} мм²")
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")


def main():
    initial, best, history = genetic_algorithm()

    print_layout("Початкове розміщення (найкраще в 1-му поколінні):", initial)
    print_layout("Оптимізоване розміщення:", best)
    print(f"Кількість поколінь: {len(history['best'])}")
    print("Координати центрів елементів:")
    for k, (x, y) in enumerate(best):
        print(f"  {k + 1:2d}: ({x:6.2f}, {y:6.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw_layout(axes[0], initial, "Початкове розміщення")
    draw_layout(axes[1], best, "Оптимізоване розміщення")
    fig.tight_layout()
    fig.savefig(folder / "layout.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(8, 4.5))
    plt.plot(history["best"], label="Найкраща хромосома популяції")
    plt.plot(history["feasible"], "--", label="Найкраще допустиме розміщення (S = 0)")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Покоління")
    plt.ylabel("F")
    plt.title("Зміна цільової функції за поколіннями")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / "objective.png", dpi=150, bbox_inches="tight")

    probs = np.round(np.arange(0.1, 0.61, 0.1), 1)
    generations, lengths = [], []
    print("\nВплив ймовірності мутації:")
    for p in probs:
        runs = [genetic_algorithm(p, seed) for seed in range(1, 6)]
        generations.append(np.mean([len(h["best"]) for _, _, h in runs]))
        lengths.append(np.mean([wire_length(z) for _, z, _ in runs]))
        print(f"  Pm = {p:.1f}: поколінь = {generations[-1]:.0f}, L = {lengths[-1]:.2f} мм")

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(probs, generations, width=0.06, color="C0", label="Кількість поколінь")
    ax1.set_xlabel("Ймовірність мутації")
    ax1.set_ylabel("Тривалість еволюції, поколінь")
    ax2 = ax1.twinx()
    ax2.plot(probs, lengths, "o-", color="C3", label="Довжина з'єднань")
    ax2.set_ylabel("Довжина з'єднань L, мм")
    fig.legend(loc="lower center", ncol=2)
    ax1.set_title("Залежність результату від ймовірності мутації")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(folder / "mutation.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
