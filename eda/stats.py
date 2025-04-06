import pandas as pd

data = pd.read_csv(
    r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\train\dow30\1d\all_train_2002_2016.csv", index_col=0)
data.drop(["Close"], inplace=True, axis=1)

label_counts = data['Label'].value_counts()
print(label_counts)

# Visualize label distribution
import matplotlib.pyplot as plt

# plt.bar(label_counts.index, label_counts.values, tick_label=['Hold', 'Buy', 'Sell'])
# plt.title("Label Distribution")
# plt.xlabel("Label")
# plt.ylabel("Count")
# plt.show()

# # Group by label and get mean indicator values
# indicator_means = data.groupby('Label').mean()
# indicator_variances = data.groupby('Label').var()
# import seaborn as sns
#
# plt.figure(figsize=(12, 6))
# sns.heatmap(indicator_means.T, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Average Indicator Values by Label")
# plt.xlabel("Label (0=Hold, 1=Buy, 2=Sell)")
#
# plt.figure(figsize=(12, 6))
# sns.heatmap(indicator_variances.T, annot=True, cmap="coolwarm", fmt=".4f")
# plt.title("Variance of Indicator Values by Label")
# plt.xlabel("Label (0=Hold, 1=Buy, 2=Sell)")
# plt.ylabel("Indicator")
# plt.show()
# plt.show()
#

# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# data = pd.read_csv(
#     r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\train\dow30\1d\all_train_2002_2016.csv", index_col=0)
# data.drop(["Close", "Symbol"], inplace=True, axis=1)
#
# # Load Data (Assuming `data` has indicators and labels)
# X = data.drop(columns=['Label'])  # Features (indicators)
# y = data['Label']  # Target (0 = Hold, 1 = Buy, 2 = Sell)
#
# # Train a Random Forest Classifier
# rf = RandomForestClassifier(n_estimators=10 , random_state=42)
# rf.fit(X, y)
#
# # Get Feature Importances
# feature_importances = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': rf.feature_importances_
# }).sort_values(by='Importance', ascending=False)
#
# # Plot Feature Importance
# # plt.figure(figsize=(12, 6))
# # sns.barplot(x='Importance', y='Feature', data=feature_importances, palette="coolwarm")
# # plt.title("Feature Importance (Random Forest)")
# # plt.xlabel("Importance Score")
# # plt.ylabel("Feature")
# # plt.show()
#
# import shap
#
# # Train a Random Forest Classifier
# rf.fit(X, y)
#
# # Use SHAP to explain feature importance
# explainer = shap.Explainer(rf, X)
# shap_values = explainer(X,check_additivity=False)
#
# # Plot Summary Plot (Global Feature Importance)
# shap.summary_plot(shap_values, X)