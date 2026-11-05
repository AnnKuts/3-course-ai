import argparse
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from network import DigitCNN

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += criterion(output, y).item() * len(y)
            correct += (output.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy", ylim=(0, 1.02))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=160)
    plt.close(fig)


def plot_examples(model, dataset, device):
    model.eval()
    mistakes = []
    with torch.inference_mode():
        for i in range(min(1000, len(dataset))):
            x, target = dataset[i]
            pred = int(model(x.unsqueeze(0).to(device)).argmax(1).item())
            if pred != int(target):
                mistakes.append((x, target, pred))
            if len(mistakes) == 8:
                break

    if not mistakes:
        return
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, (x, target, pred) in zip(axes.flat, mistakes):
        ax.imshow(x.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True: {target} | Pred: {pred}", fontsize=10)
        ax.axis("off")
    for ax in axes.flat[len(mistakes):]:
        ax.axis("off")
    fig.suptitle("Misclassified MNIST samples")
    fig.tight_layout()
    fig.savefig(RESULTS / "misclassified_digits.png", dpi=160)
    plt.close(fig)


def noisy_experiment(model, test_data, device, sigma=0.25):
    generator = torch.Generator().manual_seed(42)
    indices = list(range(min(1000, len(test_data))))
    loader = DataLoader(Subset(test_data, indices), batch_size=128)
    total, correct_clean, correct_noisy = 0, 0, 0
    examples = []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            noise = torch.randn(x.shape, generator=generator) * sigma
            noisy = (x + noise).clamp(0, 1)
            pred_clean = model(x.to(device)).argmax(1).cpu()
            pred_noisy = model(noisy.to(device)).argmax(1).cpu()
            correct_clean += (pred_clean == y).sum().item()
            correct_noisy += (pred_noisy == y).sum().item()
            total += len(y)
            if len(examples) < 5:
                for j in range(min(5 - len(examples), len(y))):
                    examples.append((x[j, 0].numpy(), noisy[j, 0].numpy(), int(y[j]), int(pred_noisy[j])))

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, (clean, noisy, label, prediction) in enumerate(examples):
        axes[0, i].imshow(clean, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Clean: {label}")
        axes[1, i].imshow(noisy, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Noisy: {prediction}")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "noise_experiment.png", dpi=160)
    plt.close(fig)
    return {
        "examples": total,
        "sigma": sigma,
        "clean_accuracy": correct_clean / total,
        "noisy_accuracy": correct_noisy / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MNIST CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    transform = transforms.ToTensor()
    dataset_train = datasets.MNIST(root=DATA, train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(root=DATA, train=False, download=True, transform=transform)
    train_data, val_data = random_split(
        dataset_train, [55_000, 5_000], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)
    test_loader = DataLoader(dataset_test, batch_size=256)

    print(f"Data: train={len(train_data)}, val={len(val_data)}, test={len(dataset_test)}")

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            running_correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)

        for key, value in [("train_loss", train_loss), ("train_acc", train_acc),
                           ("val_loss", val_loss), ("val_acc", val_acc)]:
            history[key].append(value)

        print(f"[{epoch}/{args.epochs}] loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, MODEL_DIR / "mnist_cnn.pt")

    model.load_state_dict(torch.load(MODEL_DIR / "mnist_cnn.pt", map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
    noisy = noisy_experiment(model, dataset_test, device)

    plot_history(history)
    plot_examples(model, dataset_test, device)

    metrics = {
        "epochs": args.epochs,
        "best_val_accuracy": best_val,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "noise_experiment": noisy,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Test acc: {test_acc:.4f} | Noisy acc: {noisy['noisy_accuracy']:.4f}")


if __name__ == "__main__":
    main()
