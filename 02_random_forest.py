import pandas as pd

from data_prep import (avaliar_modelo, carregar_dados, criar_pipeline,
                       split_dados)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

df = carregar_dados()

X_train, X_test, y_train, y_test = split_dados(df)

busca = GridSearchCV(
    criar_pipeline(RandomForestClassifier(random_state=42)),
    {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 5],
        "model__min_samples_leaf": [1, 5],
    },
    cv=5,
    scoring="roc_auc",
)

busca.fit(X_train, y_train)

print(f"Melhores parâmetros: {busca.best_params_}")
print(f"AUC médio (CV no treino): {busca.best_score_:.4f}")

modelo = busca.best_estimator_
avaliar_modelo("Random Forest", y_test,
               modelo.predict(X_test),
               modelo.predict_proba(X_test)[:, 1])

imp = permutation_importance(modelo, X_test, y_test,
                             scoring="roc_auc", n_repeats=10,
                             random_state=42)

tabela = pd.DataFrame({
    "feature": X_test.columns,
    "importancia": imp.importances_mean,
    "desvio": imp.importances_std,
}).sort_values("importancia", ascending=False)

print("\nImportância por permutação (queda média de AUC ao embaralhar):")
print(tabela.to_string(index=False))
