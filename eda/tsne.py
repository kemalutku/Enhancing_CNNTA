import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataset import FinanceImageDataset
import config

# --- Parameters ---
batch_size = 1024
tsne_components = 3
perplexity = 30
learning_rate = 200

# --- Load Dataset ---
dataset = FinanceImageDataset(config.train_dir, config.indicators)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- Flatten features ---
all_x = []
for _, _, x, _ in loader:
    x_flattened = x.view(x.size(0), -1)
    all_x.append(x_flattened)

all_x = torch.cat(all_x, dim=0).numpy()
labels = dataset.label_array[dataset.sequence_length:]

# --- Balance the dataset ---
hold_indices = np.where(labels == 0)[0]
buy_indices = np.where(labels == 1)[0]
sell_indices = np.where(labels == 2)[0]
min_count = min(len(buy_indices), len(sell_indices))

np.random.seed(42)
hold_downsample = np.random.choice(hold_indices, min_count, replace=False)
balanced_indices = np.concatenate([
    hold_downsample, buy_indices[:min_count], sell_indices[:min_count]
])
np.random.shuffle(balanced_indices)

X_balanced = all_x[balanced_indices]
y_balanced = labels[balanced_indices]

# --- t-SNE in 3D ---
tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
X_tsne_3d = tsne.fit_transform(X_balanced)

# --- Plot 3D ---
colors = ['blue', 'green', 'red']
labels_str = ['Hold', 'Buy', 'Sell']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i, label in enumerate([0, 1, 2]):
    idx = y_balanced == label
    ax.scatter(
        X_tsne_3d[idx, 0],
        X_tsne_3d[idx, 1],
        X_tsne_3d[idx, 2],
        c=colors[i],
        label=labels_str[i],
        s=15,
        alpha=0.6
    )

ax.set_title("3D t-SNE of Financial Indicators")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.legend()
plt.tight_layout()
plt.show()
