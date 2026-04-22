"""
Simple PyTorch Autoencoder for dimensionality reduction.
Compresses song feature vectors to a 2D latent space.
Falls back gracefully if torch is unavailable.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _Autoencoder(nn.Module if TORCH_AVAILABLE else object):
    """Encoder: input → 128 → 64 → 2 | Decoder: 2 → 64 → 128 → input"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def fit_autoencoder(
    X: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    progress_callback=None,
) -> dict:
    """
    Train autoencoder and return 2D embeddings.

    Args:
        X               : (n, d) float32 feature matrix (should be standardized)
        epochs          : training epochs
        batch_size      : mini-batch size
        lr              : learning rate
        progress_callback: optional callable(epoch, loss) for Streamlit progress

    Returns:
        {
          "X_reduced":       (n, 2) numpy array — latent 2D embeddings,
          "loss_history":    list of per-epoch reconstruction loss,
          "final_loss":      float,
          "available":       True,
        }
        or {"available": False, "reason": str} if torch is missing.
    """
    if not TORCH_AVAILABLE:
        return {
            "available": False,
            "reason": "PyTorch is not installed. Run: pip install torch",
        }

    X_f = X.astype(np.float32)
    tensor = torch.from_numpy(X_f)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _Autoencoder(X_f.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(X_f)
        loss_history.append(epoch_loss)
        if progress_callback:
            progress_callback(epoch + 1, epoch_loss)

    model.eval()
    with torch.no_grad():
        _, Z = model(tensor)
    X_reduced = Z.numpy()

    return {
        "X_reduced": X_reduced,
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "available": True,
    }
