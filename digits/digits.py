import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from itertools import islice

# reproducible
torch.manual_seed(0)

train_mnist = torchvision.datasets.MNIST(Path(__file__).parent, download=True, transform=transforms.ToTensor(), train=True)
train_loader = DataLoader(train_mnist, batch_size=64, shuffle=True)

test_mnist = torchvision.datasets.MNIST(Path(__file__).parent, download=True, transform=transforms.ToTensor(), train=False)
test_loader = DataLoader(train_mnist, batch_size=64, shuffle=True)

def perceptron(layers: list[int]):
    if len(layers) < 2:
        raise ValueError("must provide two or more layers to perceptron generator")
    return nn.Sequential(nn.Linear(layers[0], layers[1]), *[layer for n, m in zip(layers[1:], layers[2:]) for layer in [nn.ReLU(), nn.Linear(n, m)]])

model = nn.Sequential(
    nn.Flatten(),
    perceptron([784, 112, 112, 10])
)

# training setup
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 120
train_losses = []
test_losses = []

test_batches = 100

def datapoint(epoch):
    with torch.no_grad():
        # testing on train seperated without grad for clarity on small epoch sizes
        total_train_loss = 0.0
        for x, y in islice(train_loader, test_batches):
            logits_y = model.forward(x)
            loss = loss_fn(logits_y, y)

            total_train_loss += loss.item()

        total_test_loss = 0.0
        for x, y in islice(test_loader, test_batches):
            logits_y = model.forward(x)
            loss = loss_fn(logits_y, y)

            total_test_loss += loss.item()
    
    avg_train_loss = total_train_loss / min(test_batches, len(test_loader))
    train_losses.append(avg_train_loss)

    avg_test_loss = total_test_loss / min(test_batches, len(test_loader))
    test_losses.append(avg_test_loss)
    
    print(f"epoch {epoch:3d} | apx. train-loss {avg_train_loss:.4f} | apx. test-loss {avg_test_loss:.4f}")

datapoint(0)

# training loop
era_times = [0]
target_loss = float("inf")
prev_loss = float("inf")
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
    
        logits_y = model.forward(x)
        loss = loss_fn(logits_y, y)

        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    # grug learning rate tuner
    loss = total_loss / len(train_loader)
    if loss > prev_loss:
        if prev_loss < target_loss:
            era_times.append(epoch)
            learning_rate /= 2.0
            optimizer.param_groups[0]["lr"] = learning_rate
            target_loss = prev_loss
    prev_loss = loss

    # scrumptious data:
    datapoint(epoch)
era_times.append(epochs)

# full evaluation
sum_score = 0.0
for x, y in test_loader:    
    logits_y = model.forward(x)
    predictions_y = torch.softmax(logits_y, dim=-1)
    sum_score += predictions_y[range(len(y)), y].sum().item() / len(y)

score = sum_score / len(test_loader)

# yummy graph!
img_path = "./training.png"
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
for era_start, era_end, n in zip(era_times, era_times[1:], range(len(era_times))):
    if n % 2 == 0:
        plt.axvspan(era_start, era_end, color="white", alpha=0.25)
    else:
        plt.axvspan(era_start, era_end, color="grey", alpha=0.25)
plt.xlabel("Epoch")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel("Cross entropy loss")
plt.yscale("log")
plt.title("Aprox. Training and Test loss")
plt.legend()
plt.figtext(
    0.5, -0.03,
    f"avg confidence in correct class: {score:.4%}",
    ha="center", va="top",
    fontsize=10,
)
print(f"Finished training with confidence in correct class of: {score:.4%}")
plt.grid(True, alpha=0.3)
plt.savefig(img_path, dpi=120, bbox_inches="tight")
print(f"Saved loss curve to {img_path}")
