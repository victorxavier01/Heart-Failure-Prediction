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
from __future__ import annotations
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
        X, y, test_size=test_size, random_state=random_state
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