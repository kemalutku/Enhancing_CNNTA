import shap
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from dataset import FinanceImageDataset
import config

# --- Load Dataset ---
dataset = FinanceImageDataset(config.train_dir, config.indicators)
loader = DataLoader(dataset, batch_size=1024, shuffle=False)

# --- Flatten features ---
all_x = []
for _, _, x, _ in loader:
    x_flat = x.view(x.size(0), -1)
    all_x.append(x_flat)
all_x = torch.cat(all_x, dim=0).numpy()

labels = dataset.label_array[dataset.sequence_length:]

# --- Balance classes ---
hold_idx = np.where(labels == 0)[0]
buy_idx = np.where(labels == 1)[0]
sell_idx = np.where(labels == 2)[0]
min_len = min(len(buy_idx), len(sell_idx))

np.random.seed(42)
hold_sample = np.random.choice(hold_idx, min_len, replace=False)
balanced_idx = np.concatenate([hold_sample, buy_idx[:min_len], sell_idx[:min_len]])
np.random.shuffle(balanced_idx)

X = all_x[balanced_idx]
y = labels[balanced_idx]

# --- Train a simple classifier ---
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X, y)

# --- Apply SHAP ---
explainer = shap.Explainer(clf, X)
shap_values = explainer(X)

# --- SHAP summary plot ---
shap.summary_plot(shap_values, X, feature_names=config.indicators, check_additivity=False)
