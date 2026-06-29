import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img_path = "loss_curve.png"

torch.manual_seed(0)
X = torch.randn(512, 10)
y = torch.randn(512, 1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

losses = []
for epoch in range(200):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 20 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
print(f"epoch {epoch+1:3d}  loss {loss.item():.4f}")

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Training loss")
plt.grid(True, alpha=0.3)
plt.savefig(img_path, dpi=120, bbox_inches="tight")
print(f"Saved loss curve to {img_path}")
