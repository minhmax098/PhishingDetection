import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np

class PhishingDataset(Dataset):
    def __init__(self, url_sequences, handcrafted_features, labels):
        self.url_sequences = torch.tensor(url_sequences, dtype=torch.long)
        self.handcrafted_features = torch.tensor(handcrafted_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.url_sequences[idx], self.handcrafted_features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cpu'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for urls, feats, labels in train_loader:
            urls, feats, labels = urls.to(device), feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(urls, feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * urls.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for urls, feats, labels in val_loader:
                urls, feats, labels = urls.to(device), feats.to(device), labels.to(device)
                outputs = model(urls, feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * urls.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training complete. Best model saved as 'best_model.pth'.")
    return model

def evaluate_model(model, test_loader, device='cpu'):
    model = model.to(device)
    model.eval()

    all_preds_probs = []
    all_labels = []

    with torch.no_grad():
        for urls, feats, labels in test_loader:
            urls, feats, labels = urls.to(device), feats.to(device), labels.to(device)
            outputs = model(urls, feats)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds_probs = np.array(all_preds_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    preds_binary = (all_preds_probs >= 0.5).astype(int)

    metrics = {
        'Accuracy': accuracy_score(all_labels, preds_binary),
        'Precision': precision_score(all_labels, preds_binary, zero_division=0),
        'Recall': recall_score(all_labels, preds_binary, zero_division=0),
        'F1-score': f1_score(all_labels, preds_binary, zero_division=0)
    }

    # Try calculating AUCs (needs both classes present)
    try:
        metrics['ROC-AUC'] = roc_auc_score(all_labels, all_preds_probs)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds_probs)
        metrics['PR-AUC'] = auc(recall, precision)
    except ValueError:
        metrics['ROC-AUC'] = None
        metrics['PR-AUC'] = None

    return metrics, all_preds_probs, all_labels

if __name__ == "__main__":
    print("Train and Evaluation module ready.")
