import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Accuracy du Random Forest : {model.score(X_test, y_test):.4f}")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap_values_class1 = shap_values[:, :, 1]

patient_idx = 0
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_class1[patient_idx], show=False)
plt.title(f"Explication Locale SHAP - Patient {patient_idx}")
plt.tight_layout()
plt.savefig("shap_waterfall.png")
plt.close()
print("Waterfall plot sauvegardé dans shap_waterfall.png")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.title("Importance globale et directionnelle des variables (SHAP)")
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()
print("Summary plot sauvegardé dans shap_summary.png")
