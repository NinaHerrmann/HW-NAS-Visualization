import torch
from esp_ppq import TorchExecutor


@torch.no_grad()
def top1(model, loader):
    correct = total = 0
    for x, y in loader:
        out = model(x)
        logits = out[-1] if isinstance(out, (list, tuple)) else out   # (features, logits)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def graph_top1(graph, loader):
    """Accuracy of a quantized PPQ graph (simulated quantization)."""
    return top1(TorchExecutor(graph, device="cpu"), loader)


def make_autoquant_evaluate_fn(search_loader, test_loader, track):
    """Score = accuracy on the search subset. The test accuracy is only computed
    for the current best candidate and never used for selection."""
    def evaluate_fn(graph):
        executor = TorchExecutor(graph, device="cpu")
        acc = top1(executor, search_loader)
        if acc > track["eval_acc"]:
            track["eval_acc"] = acc
            track["test_acc"] = top1(executor, test_loader)
        return acc, {"top1": acc}
    return evaluate_fn