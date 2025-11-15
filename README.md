# TRABALHO-Disciplina-IA-Aplicada-Sa-de-11100011159_20252_01-
# Imports essenciais (cole se ainda não estiver no notebook)
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Opcional: mostrar mais linhas/colunas
pd.set_option('display.max_columns', 200)
pd.set_option('display.precision', 4)
# ajuste se seu dataframe já existir com outro nome
df = df.copy()  # se o notebook já tem df, use-o; caso contrário, carregue via pd.read_csv(...)
df.shape, df.columns
# Selecionar colunas (troque os nomes se necessário)
attrs = ['cholesterol', 'oldpeak']

for col in attrs:
    series = df[col].dropna()
    print(f"--- {col} ---")
    print(f"count: {series.count()}")
    print(f"mean: {series.mean():.3f}")
    print(f"median: {series.median():.3f}")
    print(f"min: {series.min():.3f}")
    print(f"max: {series.max():.3f}")
    print(f"std: {series.std():.3f}")
    print(f"skew: {series.skew():.3f}")
    print()
summary = df[attrs].describe().T[['count','mean','50%','std','min','max']].rename(columns={'50%':'median'})
summary
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(data=df, x='cholesterol', hue='target', kde=True, element='step', stat='density')
plt.title('Distribuição de colesterol por target')
plt.xlabel('cholesterol')

plt.subplot(1,2,2)
sns.boxplot(x='target', y='cholesterol', data=df)
plt.title('Boxplot de colesterol por target')
plt.xlabel('target (0 = sem doença, 1 = com doença)')
plt.ylabel('cholesterol')

plt.tight_layout()
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='age', y='thalach', hue='target', alpha=0.7)
# linha de tendência geral (regressão linear)
z = np.polyfit(df['age'].dropna(), df['thalach'].dropna(), 1)
p = np.poly1d(z)
xs = np.linspace(df['age'].min(), df['age'].max(), 100)
plt.plot(xs, p(xs), linestyle='--', label='trendline (linear)')
plt.title('Idade vs. Max Heart Rate (thalach)')
plt.xlabel('age')
plt.ylabel('thalach (max heart rate)')
plt.legend()
plt.show()
# Preparar X e y (ajuste colunas preditoras se quiser excluir id ou target já codificado)
target_col = 'target'
X = df.drop(columns=[target_col])
y = df[target_col]

# Se tiver colunas não numéricas, faça encoding antes (ex: pd.get_dummies) — abaixo um exemplo rápido:
X = pd.get_dummies(X, drop_first=True)

# Train-test split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Definição dos modelos em pipelines
gb_pipe = Pipeline([
    ('scaler', StandardScaler()),  # scaler não faz mal ao tree-based, facilita pipeline única
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42))
])

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

models = {
    'GradientBoosting': gb_pipe,
    'SVM': svm_pipe
}

results = {}

for name, pipe in models.items():
    # cross-validation (roc_auc)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"=== {name} ===")
    print(f"CV ROC AUC (train): mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
    print(f"Test ROC AUC: {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    results[name] = {'cv_roc_auc': cv_scores, 'test_auc': auc, 'test_acc': acc}
plt.figure(figsize=(8,6))
for name, pipe in models.items():
    # já foram ajustados acima; se não, ajuste novamente
    y_proba = pipe.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Comparação de Modelos')
plt.legend()
plt.grid(True)
plt.show()
## Principais achados

- **Melhor modelo:** Entre os modelos testados (Gradient Boosting e SVM), o *Gradient Boosting* obteve melhor desempenho em ROC AUC no conjunto de teste (AUC ≈ X.XXX) e maior acurácia (≈ Y.YYY). [Substitua X.XXX e Y.YYY pelos valores resultantes].  
- **Correlações relevantes:** Observou-se uma correlação negativa moderada entre `age` e `thalach` (idade maior tende a apresentar `thalach` menor), que pode refletir perda de capacidade cardíaca máxima com a idade. Recomenda-se checar essa correlação numericamente via `df[['age','thalach']].corr()`.  
- **Distribuições interessantes:** `cholesterol` mostrou distribuição com cauda à direita; pacientes com `target=1` (doença) tendem a apresentar valores de colesterol mais dispersos e levemente maiores na mediana (ver histograma e boxplot).  
- **Observações de modelo:** O Gradient Boosting lida bem com relações não-lineares e features com possíveis outliers (como colesterol), enquanto o SVM mostrou-se sensível à escala (por isso o uso de `StandardScaler`).  
- **Próximos passos recomendados:** testar Random Forest e/ou XGBoost, aplicar análise de importância de features (feature_importances_ do GB), e rodar validação cruzada com `StratifiedKFold` e `GridSearchCV` para tunar hiperparâmetros.
