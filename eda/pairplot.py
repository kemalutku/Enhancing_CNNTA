import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config

# CSV dosyasını oku
df = pd.read_csv(config.train_dir, index_col=0)

# Subset of indicators you want to analyze
indicators = config.indicators

# Optional: map numeric labels to text (for better legend readability)
label_map = {0: "Hold", 1: "Buy", 2: "Sell"}
df["LabelText"] = df["Label"].map(label_map)
min_count = df["LabelText"].value_counts().min()
balanced_df = df.groupby("LabelText").apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)


# Sample a subset if the data is too big
df = df.sample(n=500, random_state=42)

# Create pairplot
sns.pairplot(balanced_df[indicators + ["LabelText"]], hue="LabelText", corner=True, plot_kws={"alpha": 0.5})
plt.suptitle("Pairwise Indicator Distributions by Label", y=1.02)
plt.show()
