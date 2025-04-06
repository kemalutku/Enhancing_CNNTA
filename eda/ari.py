import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import config

# 1. Veriyi yükle
df = pd.read_csv(config.train_dir, index_col=0)

# 2. Gerekli kolonlar
indicators = config.indicators

# 3. Normalize et
X = df[indicators]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Gerçek etiketleri al
true_labels = df["Label"]

# 5. KMeans kümeleme
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
predicted_labels = kmeans.fit_predict(X_scaled)

# 6. ARI hesapla
ari = adjusted_rand_score(true_labels, predicted_labels)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")

# 7. (Opsiyonel) PCA ile görselleştir
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_plot = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "TrueLabel": true_labels.map({0: "Hold", 1: "Buy", 2: "Sell"}),
    "Cluster": predicted_labels
})

plt.figure(figsize=(12, 5))

# Gerçek etiketler
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="TrueLabel", palette="Set1")
plt.title("Gerçek Etiketler")

# K-Means kümeleri
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="Set2")
plt.title("K-Means Kümeleri")

plt.tight_layout()
plt.show()
