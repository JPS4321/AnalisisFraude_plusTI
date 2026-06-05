"""
evaluar_modelo.py
Compara las predicciones de un modelo contra la respuesta oficial y reporta métricas.

Uso:
    python evaluar_modelo.py <oficial.xlsx> <predicciones.xlsx>

Ejemplo:
    python evaluar_modelo.py inferencia_30pct_hibrido_final.xlsx inferencia_30pct_hibrido_con_id.xlsx

Ambos archivos deben tener columnas: transaction_id, is_fraud
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef
)


def load_and_validate(path: str, label: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"transaction_id", "is_fraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{label}] Faltan columnas: {missing}")
    df["is_fraud"] = df["is_fraud"].astype(bool)
    return df[["transaction_id", "is_fraud"]]


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python evaluar_modelo.py <oficial.xlsx> <predicciones.xlsx>")
        sys.exit(1)

    path_oficial = sys.argv[1]
    path_pred    = sys.argv[2]

    # ── Carga ──────────────────────────────────────────────────
    df_oficial = load_and_validate(path_oficial, "oficial")
    df_pred    = load_and_validate(path_pred,    "predicciones")

    # ── Merge por transaction_id ───────────────────────────────
    merged = df_oficial.merge(
        df_pred, on="transaction_id", suffixes=("_oficial", "_pred")
    )

    n_oficial = len(df_oficial)
    n_pred    = len(df_pred)
    n_matched = len(merged)
    n_missing = n_oficial - n_matched

    print_section("COBERTURA")
    print(f"  Transacciones en oficial       : {n_oficial:,}")
    print(f"  Transacciones en predicciones  : {n_pred:,}")
    print(f"  Transacciones coincidentes     : {n_matched:,}")
    if n_missing > 0:
        print(f"  ⚠  Transacciones sin predicción : {n_missing:,}")

    y_true = merged["is_fraud_oficial"].astype(int)
    y_pred = merged["is_fraud_pred"].astype(int)

    # ── Distribución ───────────────────────────────────────────
    print_section("DISTRIBUCIÓN DE CLASES")
    fraud_true = y_true.sum()
    fraud_pred = y_pred.sum()
    print(f"  Fraudes reales (oficial)       : {fraud_true:,}  ({fraud_true/len(y_true)*100:.2f}%)")
    print(f"  Fraudes predichos (modelo)     : {fraud_pred:,}  ({fraud_pred/len(y_pred)*100:.2f}%)")

    # ── Matriz de confusión ────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print_section("MATRIZ DE CONFUSIÓN")
    print(f"  {'':20s}  Pred: No Fraude   Pred: Fraude")
    print(f"  {'Real: No Fraude':20s}  {tn:>15,}   {fp:>12,}")
    print(f"  {'Real: Fraude':20s}  {fn:>15,}   {tp:>12,}")

    # ── Métricas principales ───────────────────────────────────
    accuracy   = accuracy_score(y_true, y_pred)
    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    mcc        = matthews_corrcoef(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr         = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ROC-AUC y PR-AUC (solo si hay ambas clases)
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
        pr_auc  = average_precision_score(y_true, y_pred)
    except ValueError:
        roc_auc = pr_auc = float("nan")

    print_section("MÉTRICAS PRINCIPALES")
    print(f"  Accuracy                       : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision (PPV)                : {precision:.4f}")
    print(f"  Recall (Sensitivity / TPR)     : {recall:.4f}")
    print(f"  Specificity (TNR)              : {specificity:.4f}")
    print(f"  F1-Score                       : {f1:.4f}")
    print(f"  MCC                            : {mcc:.4f}")
    print(f"  ROC-AUC                        : {roc_auc:.4f}")
    print(f"  PR-AUC (Avg Precision)         : {pr_auc:.4f}")

    print_section("TASAS DE ERROR")
    print(f"  False Positive Rate (FPR)      : {fpr:.4f}  → {fp:,} transacciones legítimas marcadas como fraude")
    print(f"  False Negative Rate (FNR)      : {fnr:.4f}  → {fn:,} fraudes no detectados")

    print_section("REPORTE DETALLADO (sklearn)")
    print(classification_report(
        y_true, y_pred,
        target_names=["No Fraude", "Fraude"],
        digits=4
    ))

    print("=" * 60)
    print("  Evaluación completa.")
    print("=" * 60)


if __name__ == "__main__":
    main()