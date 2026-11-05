
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import random_split
from network import DigitCNN

root = Path(__file__).resolve().parent

dataset = datasets.MNIST(
    root=root / "data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_data, _ = random_split(
    dataset, [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)

model = DigitCNN()
model.load_state_dict(
    torch.load(root / "model/mnist_cnn.pt",
               map_location="cpu", weights_only=True)
)
model.eval()

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

with torch.inference_mode():
    for i, ax in enumerate(axes.flat):
        image, label = train_data[i]
        pred = model(image.unsqueeze(0)).argmax(1).item()

        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"True: {label} | Pred: {pred}",
                     color="green" if pred == label else "red")
        ax.axis("off")

fig.tight_layout()
(root / "results").mkdir(exist_ok=True)
fig.savefig(root / "results/train_predictions.png", dpi=160)
plt.close(fig)

print("Saved results/train_predictions.png")
