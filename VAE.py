
# VARIATIONAL AUTOENCODER (VAE) USING MNIST

# 1. Import libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 2. Set device and output directory
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

print("Device:", device)

# 3. Load and preprocess MNIST dataset
original_dim = 784
latent_dim = 2
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# 4. Define VAE model
class VAE(nn.Module):

    def __init__(self, original_dim=784, latent_dim=2):
        super().__init__()

        # Encoder layers
        self.enc_fc1 = nn.Linear(original_dim, 256)
        self.enc_fc2 = nn.Linear(256, 64)

        # Mean and log variance
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder layers
        self.dec_fc1 = nn.Linear(latent_dim, 64)
        self.dec_fc2 = nn.Linear(64, 256)
        self.dec_out = nn.Linear(256, original_dim)

    # Encode input into mean and log variance
    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))

        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)

        return z_mean, z_logvar

    # Reparameterization trick
    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(std)

        z = z_mean + std * epsilon
        return z

    # Decode latent vector
    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))

        reconstruction = torch.sigmoid(self.dec_out(h))
        return reconstruction

    # Complete forward pass
    def forward(self, x):
        z_mean, z_logvar = self.encode(x)

        z = self.reparameterize(z_mean, z_logvar)

        reconstruction = self.decode(z)

        return reconstruction, z_mean, z_logvar


# 5. Define VAE loss function
def vae_loss(reconstruction, x, z_mean, z_logvar):

    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(
        reconstruction, x, reduction="sum"
    )

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(
        1 + z_logvar
        - z_mean.pow(2)
        - z_logvar.exp()
    )

    # Total loss
    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss


# 6. Create model and optimizer
model = VAE(original_dim, latent_dim).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3
)

epochs = 15


# 7. Train the VAE
for epoch in range(1, epochs + 1):

    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    for x, _ in train_loader:

        x = x.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        reconstruction, z_mean, z_logvar = model(x)

        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss(
            reconstruction, x, z_mean, z_logvar
        )

        # Backpropagation and weight update
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_samples += x.size(0)

    print(
        f"Epoch {epoch:2d}/{epochs} | "
        f"Loss: {total_loss / n_samples:.4f} | "
        f"Reconstruction: {total_recon / n_samples:.4f} | "
        f"KL: {total_kl / n_samples:.4f}"
    )


# 8. Generate new handwritten digits
model.eval()

n = 15
digit_size = 28

figure = np.zeros((
    digit_size * n,
    digit_size * n
))

grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)[::-1]

with torch.no_grad():

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):

            # Create a 2D latent point
            z_sample = torch.tensor(
                [[xi, yi]], dtype=torch.float32
            ).to(device)

            # Decode latent point
            x_decoded = model.decode(z_sample).cpu().numpy()

            # Convert 784 values into a 28x28 image
            digit = x_decoded[0].reshape(
                digit_size, digit_size
            )

            # Place digit in the output grid
            figure[
                i * digit_size:(i + 1) * digit_size,
                j * digit_size:(j + 1) * digit_size
            ] = digit

# Display and save generated digits
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="Greys_r")
plt.axis("off")
plt.title("Digits Generated by VAE")

plt.savefig(
    os.path.join(output_dir, "vae_generated_digits.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()


# 9. Visualize the latent space
all_z_mean = []
all_labels = []

with torch.no_grad():

    for x, y in train_loader:

        x = x.to(device)

        z_mean, _ = model.encode(x)

        all_z_mean.append(z_mean.cpu().numpy())
        all_labels.append(y.numpy())

all_z_mean = np.concatenate(all_z_mean, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Plot the latent coordinates
plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    all_z_mean[:, 0],
    all_z_mean[:, 1],
    c=all_labels,
    cmap="tab10",
    s=2
)

plt.colorbar(scatter, label="Digit label")
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.title("MNIST Latent Space")

plt.savefig(
    os.path.join(output_dir, "vae_latent_space.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close()


# 10. Final result
print("\nGenerated images saved in:", output_dir)
print("Result: VAE training, digit generation and latent-space visualization completed.")
