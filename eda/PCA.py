import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
from dataset import FinanceImageDataset
import config
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters ---
pca_components = 15  # Number of components to keep
batch_size = 1024  # Adjust based on your memory

# --- Load Dataset ---
dataset = FinanceImageDataset(config.train_dir, config.indicators)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- Collect all data for PCA ---
all_x = []

for _, _, x, _ in loader:
    # Flatten each sample from (batch, seq_len, n_features) -> (batch, seq_len * n_features)
    x_flattened = x.view(x.size(0), -1)
    all_x.append(x_flattened)

# Stack all into one matrix for PCA
all_x = torch.cat(all_x, dim=0).numpy()

# --- Apply PCA ---
pca = PCA(n_components=pca_components)
all_x_pca = pca.fit_transform(all_x)

# --- (Optional) Explained Variance ---
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by each component: {explained_variance}')
print(f'Cumulative explained variance: {np.cumsum(explained_variance)}')

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. PCA Components')
plt.grid(True)
plt.show()

labels = dataset.label_array[dataset.sequence_length:]  # align with PCA input
colors = ['gray', 'green', 'red']  # hold, buy, sell

# --- 2D Visualization ---
plt.figure(figsize=(8, 6))
for i, label in enumerate([0, 1, 2]):
    plt.scatter(all_x_pca[labels == label, 0],
                all_x_pca[labels == label, 1],
                alpha=0.6,
                label=['Hold', 'Buy', 'Sell'][i],
                c=colors[i])

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA of Financial Features')
plt.legend()
plt.grid(True)
plt.show()
