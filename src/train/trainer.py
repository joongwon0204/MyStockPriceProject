import torch
import torch.nn as nn

def train_one_epoch(model, loader, opt, device="cuda"):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()

    return total_loss / len(loader.dataset), correct / total

@torch.no_grad()
def eval_one_epoch(model, loader, device="cuda"):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()

    return total_loss / len(loader.dataset), correct / total