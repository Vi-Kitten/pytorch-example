import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img_path = "loss_curve.png"

# reproducible, very based, much approve.
torch.manual_seed(0)

# some incredibly business integral data, wouldn't want this being leaked to the public.
x = torch.randn(512, 10)
y = torch.randn(512, 1)

# create model from sequential layers.
# - 10 input
# - 64 hidden
# - 1  output
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

# not justified by log scoring or information theory, cringe!
loss_fn = nn.MSELoss()

# I guess this optimiser is called Adam, seems pretty heavy for giving me something like MSE :p, what do I know tho.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

epochs = 200
losses = []
for epoch in range(epochs):
    # resets gradients, kinda feels like clearing a canvas, fun...
    # how terrifyingly cool that this seems to be optional!
    optimizer.zero_grad()

    # "pred" could mean predicate, I was so confused for like 15 minutes...
    # this is predicted though I get it, what an interesting object this seems to make.
    pred = model(x)
    loss = loss_fn(pred, y)

    # undocumented again, but I know what this is and must be.
    # backproppers!
    # Jacobians go brrrrrrrrr
    loss.backward()

    # "Perform a single optimisation step", no shit dude, that's the function name.
    # I assume this is general, and that for normal gradient descent this is just stepping with the given gradients.
    optimizer.step()

    # scrumptious data:
    losses.append(loss.item())
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

# yummy graph!
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Training loss")
plt.grid(True, alpha=0.3)
plt.savefig(img_path, dpi=120, bbox_inches="tight")
print(f"Saved loss curve to {img_path}")
