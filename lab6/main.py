from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

n = 600
duration = rng.uniform(10, 300, n)
completeness = rng.uniform(0, 100, n)

efficiency = (
    0.81 * completeness
    + 15 * np.exp(-0.5 * ((duration - 128) / 65) ** 2)
    - 0.14 * np.maximum(duration - 190, 0)
    - 0.10 * np.maximum(55 - duration, 0)
    + rng.normal(0, 2.2, n)
)
efficiency = np.clip(efficiency, 0, 100)

folder = Path(__file__).resolve().parent
np.savetxt(folder / "training_data.dat",
           np.column_stack((duration, completeness, efficiency)), fmt="%.4f")

indices = rng.permutation(n)
train_ids, test_ids = indices[:480], indices[480:]
X = np.column_stack((duration / 300, completeness / 100)).astype("float32")
y = (efficiency / 100).astype("float32")
X_train = torch.tensor(X[train_ids])
y_train = torch.tensor(y[train_ids])
X_test = torch.tensor(X[test_ids])
y_test = torch.tensor(y[test_ids])

class ANFIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.centers = nn.Parameter(torch.tensor([
            [0.12, 0.50, 0.88],
            [0.12, 0.50, 0.88],
        ]))
        self.log_sigma = nn.Parameter(torch.full((2, 3), float(np.log(0.3))))
        self.p = nn.Parameter(torch.randn(9, 3) * 0.1)

    def forward(self, inputs):
        sigma = self.log_sigma.exp()
        mu = torch.exp(-0.5 * ((inputs[:, :, None] - self.centers) / sigma) ** 2)
        w = (mu[:, 0, :, None] * mu[:, 1, None, :]).reshape(-1, 9)
        features = torch.cat((torch.ones((len(inputs), 1)), inputs), dim=1)
        rule_outputs = features @ self.p.T
        return (w * rule_outputs).sum(1) / (w.sum(1) + 1e-12)

model = ANFIS()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_function = nn.MSELoss()
epochs = 400
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = loss_function(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        model.centers.clamp_(0.0, 1.0)
        model.log_sigma.clamp_(float(np.log(0.08)), float(np.log(0.65)))
    losses.append(loss.item() * 10000)  # MSE в балах²
    if (epoch + 1) % 100 == 0:
        print(f"Епоха {epoch + 1}: MSE = {losses[-1]:.3f} балів²")

model.eval()
with torch.no_grad():
    predicted = model(X_test).numpy() * 100
    sample = torch.tensor([[120 / 300, 85 / 100]], dtype=torch.float32)
    new_prediction = float(model(sample).item() * 100)

actual = efficiency[test_ids]
predicted = np.clip(predicted, 0, 100)
mae = float(np.mean(np.abs(actual - predicted)))
rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
r2 = float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
print(f"\nНавчальних прикладів: {len(train_ids)}; тестових: {len(test_ids)}")
print(f"MAE = {mae:.2f} бала; RMSE = {rmse:.2f} бала; R² = {r2:.3f}")
print(f"Новий дзвінок: 120 с, повнота 85% => ефективність {float(np.clip(new_prediction, 0, 100)):.1f}/100")

terms = ["низька", "середня", "висока"]
with torch.no_grad():
    centers_order = np.argsort(model.centers.numpy(), axis=1)
term_names: list[list[str]] = [["" for _ in range(3)] for _ in range(2)]
for k in range(2):
    for position, mf_index in enumerate(centers_order[k]):
        term_names[k][mf_index] = terms[position]
with torch.no_grad():
    for i, (a, b) in enumerate(product(range(3), repeat=2)):
        p0, p1, p2 = (float(value) for value in model.p[i].numpy())
        print(f"Правило {i+1}: ЯКЩО тривалість {term_names[0][a]}, а повнота {term_names[1][b]}, "
              f"ТО y = {100*p0:+.2f} {p1/3:+.3f}*сек {p2:+.3f}*повнота%")

plt.rcParams["font.family"] = "DejaVu Sans"

fig1, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig1.suptitle("Структура гібридної нейро-нечіткої мережі ANFIS", fontsize=15)

x_in, x_mf, x_rule, x_then, x_sum, x_out = .06, .24, .45, .64, .81, .94
in_y = [.76, .24]
mf_y = [[.91, .79, .67], [.33, .21, .09]]
rule_y = np.linspace(.90, .10, 9)

for label_x, label in [(x_in, "Входи"), (x_mf, "Фазифікація"),
                 (x_rule, "9 нечітких\nправил"), (x_then, "Наслідки\nСугено"),
                 (x_sum, "Агрегація"), (x_out, "Вихід")]:
    ax.text(label_x, .98, label, ha="center", va="top", fontsize=10, fontweight="bold")


def draw_node(node_x, node_y, text, color="#edf2f7", size=9):
    ax.text(node_x, node_y, text, ha="center", va="center", fontsize=size,
            bbox=dict(boxstyle="round,pad=0.36", fc=color, ec="#66788c"), zorder=3)

def connect(start_x, start_y, end_x, end_y):
    ax.plot([start_x, end_x], [start_y, end_y], color="#8d99a8", lw=.7, alpha=.5, zorder=1)

for i, (node_y, node_name) in enumerate(zip(in_y, ["Тривалість\n(с)", "Повнота\n(%)"])):
    draw_node(x_in, node_y, node_name, color="#e7f0fc")
    for j, mf_y_position in enumerate(mf_y[i]):
        connect(x_in+.03, node_y, x_mf-.03, mf_y_position)
        draw_node(x_mf, mf_y_position, ["Низька", "Середня", "Висока"][j], color="#e6f5ec", size=8)

for i, (a, b) in enumerate(product(range(3), repeat=2)):
    rule_y_position = rule_y[i]
    connect(x_mf+.035, mf_y[0][a], x_rule-.027, rule_y_position)
    connect(x_mf+.035, mf_y[1][b], x_rule-.027, rule_y_position)
    draw_node(x_rule, rule_y_position, f"R{i+1}", color="#fdf0dc", size=8)
    connect(x_rule+.02, rule_y_position, x_then-.028, rule_y_position)
    draw_node(x_then, rule_y_position, f"f{i+1}(x)", color="#f5eafd", size=8)
    connect(x_then+.035, rule_y_position, x_sum-.035, .68)
    connect(x_rule+.025, rule_y_position, x_sum-.035, .30)

draw_node(x_sum, .68, "Σ(w·f)", color="#edf2f7")
draw_node(x_sum, .30, "Σw", color="#edf2f7")
draw_node(x_out, .49, "y = Σ(w·f) / Σw\n0–100 балів", color="#e0f4f2")
connect(x_sum+.035, .68, x_out-.06, .49)
connect(x_sum+.035, .30, x_out-.06, .49)
fig1.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
fig1.savefig(folder / "01_structure.png", dpi=180)
plt.show()  # Закрийте вікно, щоб перейти до наступного рисунка.

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.6))
fig2.suptitle("Навчені функції приналежності вхідних змінних", fontsize=14)
x_axis = np.linspace(0, 1, 300)
with torch.no_grad():
    centers = model.centers.numpy()
    sigmas = model.log_sigma.exp().numpy()
for k, (ax, title, scale) in enumerate([
    (axes2[0], "Тривалість дзвінка", 300),
    (axes2[1], "Повнота інформації", 100),
]):
    for label_index, mf_index in enumerate(centers_order[k]):
        membership_values = np.exp(-0.5 * ((x_axis - centers[k, mf_index]) / sigmas[k, mf_index]) ** 2)
        ax.plot(x_axis * scale, membership_values, label=terms[label_index])
    ax.set(title=title, xlabel="Секунди" if k == 0 else "%",
           ylabel="Ступінь приналежності")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
fig2.tight_layout()
fig2.savefig(folder / "02_memberships.png", dpi=180)
plt.show()

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.6))
fig3.suptitle("Навчання та перевірка ANFIS", fontsize=14)
axes3[0].plot(range(1, epochs + 1), losses)
axes3[0].set(title="Зміна похибки під час навчання", xlabel="Епоха", ylabel="MSE, балів²")
axes3[0].grid(alpha=.3)
axes3[1].scatter(actual, predicted, alpha=.6, s=18)
axes3[1].plot([0, 100], [0, 100], "--", color="gray", label="Ідеальний прогноз")
axes3[1].set(title=f"Тестова вибірка: MAE = {mae:.2f} бала",
             xlabel="Еталонна оцінка", ylabel="Прогноз ANFIS",
             xlim=(0, 100), ylim=(0, 100))
axes3[1].legend(fontsize=9)
axes3[1].grid(alpha=.3)
fig3.tight_layout()
fig3.savefig(folder / "03_training_and_test.png", dpi=180)
plt.show()

D, C = np.meshgrid(np.linspace(10, 300, 80), np.linspace(0, 100, 80))
surface_input = torch.tensor(
    np.column_stack((D.ravel() / 300, C.ravel() / 100)), dtype=torch.float32
)
with torch.no_grad():
    surface = model(surface_input).numpy().reshape(D.shape) * 100
fig4, ax4 = plt.subplots(figsize=(8, 5.5))
heat = ax4.contourf(D, C, np.clip(surface, 0, 100), levels=20,
                    cmap="viridis", vmin=0, vmax=100)
ax4.set(title="Поверхня нечіткого виводу Сугено",
        xlabel="Тривалість дзвінка, с", ylabel="Повнота інформації, %")
fig4.colorbar(heat, ax=ax4, label="Прогнозована ефективність, бали")
fig4.tight_layout()
fig4.savefig(folder / "04_sugeno_surface.png", dpi=180)
plt.show()
