import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Optional
import logging


def evaluate_latent_classification(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Evaluates the quality of embeddings by training a simple classifier
    on the frozen embeddings produced by the model.

    Args:
        model: The trained model that outputs embeddings
        dataloader: A DataLoader that yields (x, label) pairs
        device: The torch device to use
        logger: Optional logger for diagnostics

    Returns:
        Accuracy score (float)
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            emb = model(x)  # [B, D]
            all_embeddings.append(emb.cpu())
            all_labels.append(y)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(embeddings, labels)
    preds = clf.predict(embeddings)

    acc = float(accuracy_score(labels, preds))

    if logger:
        logger.info(f"Diagnostic classifier accuracy: {acc:.4f}")

    return acc
