"""Preparação dos dados para os modelos.

Como queremos usar o mesmo split e pipelines nos 3 modelos, vou criar um script que fique responsável pela preparação dos dados.
Isso é ideal pois queremos comparar os modelos com os mesmos dados.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "heart.csv"
TARGET = "HeartDisease"
RANDOM_STATE = 42
TEST_SIZE = 0.2

COLUNAS_NUMERICAS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
COLUNAS_CATEGORICAS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
COLUNAS_FEATURES = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS

def carregar_dados() -> pd.DataFrame:
    """Carregar o CSV e valida a integridade (Nans, dupliatas etc...)"""
    df = pd.read_csv(DATA)
    checar_integridade(df)

    return df

def checar_integridade(df: pd.DataFrame) -> None:
    valores_ausentes = int(df[COLUNAS_FEATURES + [TARGET]].isna().sum().sum())
    if valores_ausentes:
        raise ValueError(f"Existem {valores_ausentes} valores ausentes no dataset.")

    duplicatas = int(df.duplicated(subset=COLUNAS_FEATURES).sum())
    if duplicatas:
        raise ValueError(f"Existem {duplicatas} duplicatas no dataset.")
    print(f"[data_prep] {len(df)} amostras | {len(COLUNAS_FEATURES)} features")
    print(f"[data_prep] proporção da classe 1: {(df[TARGET] == 1).mean():.3f}")

def split_dados(
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split 80/20 dos dados em treino e teste"""
    X = df[COLUNAS_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def criar_preprocessador() -> ColumnTransformer:
    """One-hot e StandardScaler"""

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS),
            ("num", StandardScaler(), COLUNAS_NUMERICAS)
        ],
        remainder="drop"
    )

def criar_pipeline(modelo) -> Pipeline:
    """O modelo SEMPRE deverá entrar aqui."""
    return Pipeline([
        ("prep", criar_preprocessador()),
        ("model", modelo)
    ])

def pegar_nomes_features(pipeline: Pipeline) -> list[str]:
    """Nomes das features após o preprocessamento (OneHotEncoder e StandardScaler)"""
    return list(pipeline.named_steps['prep'].get_feature_names_out())

def formatar_matriz_confusao(cm: np.ndarray) -> str:
    lines = [" " * 10 + f"{'pred 0':>9} {'pred 1':>9}"]
    for i in range(cm.shape[0]):
        lines.append(f"{'true ' + str(i):>10} {cm[i][0]:>9d} {cm[i][1]:>9d}")
    return "\n".join(lines)

def avaliar_modelo(name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Relatório padrão no test set"""

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f} (metrica principal)")
    print(f"F1 (cl. 1): {metrics['f1']:.4f}")
    print("\nClassification report (classe 1 = doença):")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Matriz de confusão:")
    print(formatar_matriz_confusao(np.asarray(confusion_matrix(y_true, y_pred))))

    return metrics