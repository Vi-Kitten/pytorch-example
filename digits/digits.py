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
    nn.Conv2d(in_channels=1, out_channels=4, kernel_size=8, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    perceptron([32, 12, 10])
)
model_params = sum(p.numel() for p in model.parameters())

# training setup
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 40
train_losses = []
test_losses = []

print(f"training model with {model_params} parameters")

# training loop
era_times = [0]
target_loss = float("inf")
prev_loss = float("inf")
for epoch in range(epochs):
    
    # test data over time
    with torch.no_grad():
        total_test_loss = 0.0
        for x, y in test_loader:
            logits_y = model.forward(x)
            loss = loss_fn(logits_y, y)

            total_test_loss += loss.item()
    
    test_loss = total_test_loss / len(test_loader)
    test_losses.append(test_loss)

    # gradiend descent
    total_train_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
    
        logits_y = model.forward(x)
        loss = loss_fn(logits_y, y)

        total_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()


    loss = total_train_loss / len(train_loader)
    train_losses.append(loss)
    
    # grud learning rate tuner
    if loss > prev_loss:
        if prev_loss < target_loss:
            era_times.append(epoch)
            learning_rate /= 2.0
            optimizer.param_groups[0]["lr"] = learning_rate
            target_loss = prev_loss
    prev_loss = loss

    # scrumptious data:    
    print(f"epoch {epoch:3d} | rolling-train-loss {loss:.4f} | prior-test-loss {test_loss:.4f} | new-learning-rate {learning_rate}")

era_times.append(epochs-1)

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
        plt.axvspan(era_start, era_end, color="white", alpha=0.125)
    else:
        plt.axvspan(era_start, era_end, color="grey", alpha=0.125)
plt.xlabel("Epoch")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel("Cross entropy loss")
plt.yscale("log")
plt.title(f"Training and Test loss for {model_params} parameter model")
plt.legend()
plt.figtext(
    0.5, -0.03,
    f"avg confidence in correct class: {score:.4%}",
    ha="center", va="top",
    fontsize=10,
)
print(f"Finished training {model_params} parameter model with confidence in correct class of: {score:.4%}")
plt.grid(True, alpha=0.3)
plt.savefig(img_path, dpi=120, bbox_inches="tight")
print(f"Saved loss curve to {img_path}")
