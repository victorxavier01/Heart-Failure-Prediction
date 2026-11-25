# Heart Failure Prediction

Experimento comparando 3 algoritmos de classificação no dataset de falha cardíaca (`heart.csv`): **Regressão Logística**, **Random Forest** e **XGBoost**.

O objetivo não é só "treinar um modelo": é compará-los sob um protocolo **idêntico** (mesmo split, mesmo pré-processamento, mesmas métricas), para que a diferença entre os resultados isole o algoritmo.

## Dataset

| | |
|---|---|
| Amostras | 918 |
| Features | 11 (6 numéricas + 5 categóricas) |
| Alvo | `HeartDisease` (binária) — 55% positivos |
| Faltas / duplicatas | nenhuma (checada em `data_prep.py`) |

## Protocolo (o que garante a comparação justa)

- **Split**: 80/20 estratificado (`random_state=42`), único e compartilhado pelos 3 modelos
- **Pipeline compartilhado** (`data_prep.py`): One-Hot nas 5 categóricas (→ 14 dummies) + `StandardScaler` nas numéricas, ajustado **somente no treino**
- **Seleção de hiperparâmetros**: `GridSearchCV` (cv=5, AUC) apenas sobre o **treino**
- **Test set**: intocado até a única avaliação final
- **Métricas**: ROC-AUC (principal), accuracy, precision/recall/F1, matriz de confusão

## Como rodar

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python data_prep.py           # smoke test do split/pipeline (não treina nada)
python 01_reg_logistica.py    # Passo 1
python 02_random_forest.py    # Passo 2
python 03_xgboost.py          # Passo 3
```

## Resultados (mesmo test set de 184 linhas)

| Algoritmo | ROC-AUC | Accuracy | F1 (cl. 1) | Erros (FP/FN) | Hiperparâmetros escolhidos |
|---|---|---|---|---|---|
| **Regressão Logística** | **0.9323** | 0.8913 | 0.9048 | 13/7 | C=0.1 |
| Random Forest | 0.9311 | 0.9022 | 0.9143 | 12/6 | 400 árvores, max_depth=None, min_samples_leaf=5 |
| XGBoost | 0.9217 | 0.8696 | 0.8868 | 16/8 | lr=0.05, max_depth=3, 56 árvores (early stopping) |

**Veredito**: a Regressão Logística venceu. O dataset é quase linearmente separável — as árvores tiveram pouco a acrescentar (RF empatou tecnicamente; XGBoost ficou levemente atrás, com early stopping conservador).

### Principais achados

- **ST_Slope é a feature dominante** — consenso dos 3 modelos (coeficiente `ST_Slope_Up −0.97` na LR, importância 0.105 na RF, 0.115 no XGB). Traçado ST normal (ascendente) reduz as chances de doença em ~62% (e^−0.97 ≈ 0.38); traçado plano multiplica as chances por ~2.4×.
- `Age`, `MaxHR`, `RestingBP` e `RestingECG` são quase ignoradas por todos os modelos.
- **Grids em planície**: nos 3 modelos o vencedor ficou dentro do ruído do CV em relação ao 2º lugar — neste dataset, a força de regularização mal importa.
- **Lição do XGBoost**: o erro do boosting tem formato de "U" — sem early stopping o modelo memoriza o treino (AUC 0.999); com valid set de 147 linhas, o ponto de parada é a decisão mais sensível.

## Estrutura

| Arquivo | Papel |
|---|---|
| `data_prep.py` | Protocolo compartilhado: carga, checagem de integridade, split estratificado, pipeline, métricas |
| `01_reg_logistica.py` | Passo 1 — Regressão Logística (grid do C + top coeficientes) |
| `02_random_forest.py` | Passo 2 — Random Forest (grid de 3 parâmetros + importância por permutação) |
| `03_xgboost.py` | Passo 3 — XGBoost (grid lr×depth + early stopping em valid set fatiado do treino) |
| `heart.csv` | Dataset |

## Notas

- Reprodutível: todas as seeds fixas (`RANDOM_STATE = 42` em `data_prep.py`).
- O early stopping do XGBoost pode ter parado um pouco cedo: um rodapé diagnóstico com 500 árvores (sem parada) atingiu AUC 0.9286 no test — o teto dessa família de modelos neste dado é ~0.93, o que não altera o veredito.
