from data_prep import (avaliar_modelo, carregar_dados, criar_pipeline,
                       pegar_nomes_features, split_dados)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


df = carregar_dados()

X_train, X_test, y_train, y_test = split_dados(df)

busca = GridSearchCV(
        criar_pipeline(LogisticRegression(max_iter=1000)),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="roc_auc",
    )

busca.fit(X_train, y_train)

print(f"Melhor C: {busca.best_params_['model__C']} "
          f"| AUC médio (CV no treino): {busca.best_score_:.4f}")

modelo = busca.best_estimator_

avaliar_modelo("Logistic Regression", y_test, modelo.predict(X_test), modelo.predict_proba(X_test)[:, 1])

nomes = pegar_nomes_features(modelo)
coefs = modelo.named_steps['model'].coef_[0]

print("\nTop coeficientes |beta| (classe 1 = doença):")

for nome, c in sorted(zip(nomes, coefs), key=lambda t: abs(t[1]), reverse=True)[:10]:
    print(f"  {nome:>28}  {c:+.4f}")