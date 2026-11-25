import pandas as pd

from data_prep import (RANDOM_STATE, avaliar_modelo, carregar_dados,
                       criar_preprocessador, criar_pipeline, split_dados)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

df = carregar_dados()

X_train, X_test, y_train, y_test = split_dados(df)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train,
    random_state=RANDOM_STATE
)

busca = GridSearchCV(
    criar_pipeline(XGBClassifier(n_estimators=500, random_state=42)),
    {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    },
    cv=5,
    scoring="roc_auc",
)

busca.fit(X_tr, y_tr)

print(f"Melhores parâmetros: {busca.best_params_}")
print(f"AUC médio (CV no treino): {busca.best_score_:.4f}")


prep = criar_preprocessador()
prep.fit(X_tr)
X_val_prep = prep.transform(X_val)

modelo = criar_pipeline(XGBClassifier(
    n_estimators=500,
    learning_rate=busca.best_params_["model__learning_rate"],
    max_depth=busca.best_params_["model__max_depth"],
    early_stopping_rounds=30,
    eval_metric="auc",
    random_state=42,
))


modelo.fit(X_tr, y_tr,
           model__eval_set=[(X_val_prep, y_val)],
           model__verbose=False)

n_arvores = modelo.named_steps["model"].best_iteration + 1
print(f"Early stopping: melhor ponto na árvore nº {n_arvores} (teto: 500)")

avaliar_modelo("XGBoost", y_test,
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
