import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config

# CSV dosyasını oku
df = pd.read_csv(config.train_dir, index_col=0)

# Eğer Label zaten 0, 1, -1 değilse, dönüştür:
df["Label"] = df["Label"].map({1: 1, 0: 0, 2: -1})

# Korelasyon analizine dahil edilecek indikatörler
indicators = config.indicators

# İlgili kolonları al
df_corr = df[indicators + ["Label"]]

# Korelasyon matrisini hesapla
corr = df_corr.corr()

# Sadece Label ile korelasyonları görselleştir
plt.figure(figsize=(10, 8))
sns.heatmap(corr[["Label"]].drop("Label"), annot=True, cmap="coolwarm", center=0)
plt.title("Label and Indicator Correlation", fontsize=14)
plt.tight_layout()
plt.savefig("label_korelasyon_heatmap.png")  # İsteğe bağlı: görseli kaydet
plt.show()
