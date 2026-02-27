import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"Accuracy de la Régression Logistique : {accuracy_score(y_test, y_pred):.4f}")

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': coefficients
})
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=True)

plt.figure(figsize=(10, 8))
colors = ['red' if c < 0 else 'blue' for c in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'][-15:], feature_importance['Coefficient'][-15:], color=colors[-15:])
plt.xlabel('Valeur du Coefficient (β)')
plt.title('Top 15 - Importance des variables (Régression Logistique)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig("glassbox_coefficients.png")
print("Graphique sauvegardé dans glassbox_coefficients.png")
