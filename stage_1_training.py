import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DATA_PATH = "ecg_train.npy"  
FAKE_DATA_PATH = "ecg_fake.npy" 
# ==============================================================================
# 1. IMPORT YOUR MODELS
# ==============================================================================

class Discriminator(nn.Module):
    def __init__(self, seq_length=3000, dropout=0.3):
        super(Discriminator, self).__init__()
        self.seq_length = seq_length
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=25, stride=2, padding=12),
            nn.LeakyReLU(0.2), nn.Dropout(dropout))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=25, stride=2, padding=12), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), nn.Dropout(dropout))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=25, stride=2, padding=12), nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2), nn.Dropout(dropout))
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=25, stride=2, padding=12), nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), nn.Dropout(dropout))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=25, stride=2, padding=12), nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2), nn.Dropout(dropout))
        self.flatten_size = 94 * 512
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(self.flatten_size, 1), nn.Sigmoid())

    def forward(self, x):
        if x.shape[-1] == 1: x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x)
        return x

# ==============================================================================
# 2. HYPERPARAMETERS
# ==============================================================================

# Training Hyperparameters
LEARNING_RATE = 0.0002
BETA1 = 0.5  # Recommended for Adam optimizer in GANs
BATCH_SIZE = 100
NUM_EPOCHS = 50 # Adjust as needed
   

# ==============================================================================
# 3. DATA LOADING
# ==============================================================================

# Load the numpy arrays
real_ecg_data = np.load(REAL_DATA_PATH)
fake_ecg_data = np.load(FAKE_DATA_PATH)

# Convert to PyTorch Tensors
real_ecg_tensor = torch.from_numpy(real_ecg_data).float()
fake_ecg_tensor = torch.from_numpy(fake_ecg_data).float()

if real_ecg_tensor.ndim == 2:
    real_ecg_tensor = real_ecg_tensor.unsqueeze(-1)  # (N, 3000) -> (N, 3000, 1)
# Create TensorDatasets
real_dataset = TensorDataset(real_ecg_tensor)
fake_dataset = TensorDataset(fake_ecg_tensor)

# Create DataLoaders
real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True)
fake_loader = DataLoader(fake_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"    - Real data shape: {real_ecg_tensor.shape}")
print(f"    - Fake data shape: {fake_ecg_tensor.shape}")

# ==============================================================================
# 4. INITIALIZE MODEL, LOSS, AND OPTIMIZER
# ==============================================================================

D = Discriminator(seq_length=3000, dropout=0.3).to(DEVICE)

# Loss Function: Binary Cross-Entropy
criterion = nn.BCELoss()

# Optimizers
optimizerD = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Labels for real and fake data
real_label = 1.0
fake_label = 0.0

# ==============================================================================
# STAGE 1: TRAINING THE DISCRIMINATOR
# ==============================================================================

D.train()  # Set discriminator to training mode

for epoch in range(NUM_EPOCHS):
    # Use zip_longest to handle datasets of potentially different lengths
    # It will loop until the longer dataset is exhausted.
    from itertools import zip_longest
    
    # Track loss for this epoch
    epoch_loss_d = 0.0
    num_batches = 0

    # We iterate through both real and fake data loaders simultaneously
    for i, (real_batch, fake_batch) in enumerate(zip_longest(real_loader, fake_loader)):
        
        # -----------------------------------------------------
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # -----------------------------------------------------
        D.zero_grad()
        
        # --- Train with all-real batch ---
        if real_batch is not None:
            real_data = real_batch[0].to(DEVICE)
            b_size = real_data.size(0)
            
            # Create labels for the real data (all 1s)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
            
            # Forward pass real batch through D
            output = D(real_data).view(-1)
            
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
        else:
            errD_real = 0
            D_x = 0

        # --- Train with all-fake batch ---
        if fake_batch is not None:
            fake_data = fake_batch[0].to(DEVICE)
            b_size = fake_data.size(0)

            # Create labels for the fake data (all 0s)
            label = torch.full((b_size,), fake_label, dtype=torch.float, device=DEVICE)
            
            # Classify all fake batch with D
            output = D(fake_data).view(-1)
            
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            
            # Calculate the gradients for this batch, accumulated with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
        else:
            errD_fake = 0
            D_G_z1 = 0

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        
        # Update D
        if real_batch is not None or fake_batch is not None:
            optimizerD.step()
            epoch_loss_d += errD.item()
            num_batches += 1


    # Print statistics for the epoch
    avg_loss_d = epoch_loss_d / num_batches
    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss_D: {avg_loss_d:.4f} | D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f}")


