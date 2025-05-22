from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from pathlib import Path

def evaluate_embedding_quality(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    run_dir: Path,
    logger,
    epoch: int,
    test_size: float = 0.2,
    random_state: int = 42
) -> float:
    """
    Train a simple classifier (RandomForest) on the embeddings and save accuracy.

    Returns:
        float: accuracy on held-out set
    """
    X = embeddings.cpu().numpy()
    y = labels.cpu().numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Log
    logger.info(f"[Epoch {epoch}] Diagnostic classifier accuracy: {acc:.4f}")

    # Save to CSV
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    acc_path = metrics_dir / "accuracy.csv"

    header = not acc_path.exists()
    with open(acc_path, "a") as f:
        if header:
            f.write("epoch,accuracy\n")
        f.write(f"{epoch},{acc:.4f}\n")

    return acc
