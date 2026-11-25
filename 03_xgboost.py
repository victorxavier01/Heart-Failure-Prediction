"""Passo 3: XGBoost.

A novidade central: o erro do boosting tem formato de "U".
Poucas árvores = subaprende; muitas árvores = memoriza o treino.
Por isso o n_estimators NÃO entra no grid: quem decide quantas árvores
usar é o EARLY STOPPING, monitorando um valid set.

REGRA DO EARLY STOPPING: o valid set é fatiado do TREINO (20%).
O test set continua intocado até a única avaliação final.

Fluxo em 2 etapas:
1. GridSearchCV escolhe learning_rate x max_depth (n_estimators é só o teto)
2. O vencedor treina com early stopping; o valid set decide "quando parar"
"""

import pandas as pd

from data_prep import (RANDOM_STATE, avaliar_modelo, carregar_dados,
                       criar_preprocessador, criar_pipeline, split_dados)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

df = carregar_dados()

X_train, X_test, y_train, y_test = split_dados(df)

# O valid set: 20% do TREINO (o test segue intocado).
# X_tr = bloco de seleção/treino; X_val = decide a parada do boosting.
# Custo: o modelo final treina em 587 linhas, não 734 — o preço do
# early stopping limpo (nada do valid vaza para o treino).
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train,
    random_state=RANDOM_STATE
)

# Etapa 1: learning_rate x max_depth.
# - learning_rate: tamanho do passo de cada árvore (menor = mais
#   conservador, precisa de mais árvores para chegar ao mesmo lugar)
# - max_depth: 3-5 é a filosofia do boosting (árvores RASAS:
#   "aprendizes fracos" que se combinam, não especialistas)
# - n_estimators=500 é só o TETO; o early stopping decide o número real
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

# Etapa 2: early stopping. Detalhe que pega gente: o Pipeline do sklearn
# NÃO transforma o eval_set sozinho. Por isso transformamos X_val com um
# preprocessador ajustado SÓ no X_tr (mesmo estado do preprocessor do
# pipeline final, que também ajusta em X_tr).
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

# Em versões recentes do sklearn, o Pipeline valida os parâmetros do fit e
# encaminha para o passo certo usando o formato stepname__parametro:
# "model__eval_set" vai direto para o fit do XGBClassifier.
modelo.fit(X_tr, y_tr,
           model__eval_set=[(X_val_prep, y_val)],
           model__verbose=False)

n_arvores = modelo.named_steps["model"].best_iteration + 1
print(f"Early stopping: melhor ponto na árvore nº {n_arvores} (teto: 500)")

avaliar_modelo("XGBoost", y_test,
               modelo.predict(X_test),
               modelo.predict_proba(X_test)[:, 1])

# Pergunta 3: mesma ferramenta da RF (importância por permutação sobre AUC)
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
