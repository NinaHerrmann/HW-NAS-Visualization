import torch
import torch.nn as nn


class ExportWrapper(nn.Module):
    """Same computation as xautodl's TinyNetwork.forward, but with a
    batch-size independent flatten (ONNX Flatten instead of Reshape([1, -1]))."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        feature = n.stem(x)
        for cell in n.cells:
            feature = cell(feature)
        out = n.lastact(feature)
        out = n.global_pooling(out)
        out = torch.flatten(out, 1)          # was: out.view(out.size(0), -1)
        logits = n.classifier(out)
        return out, logits


def export_onnx(network, onnx_file, input_shape):
    model = ExportWrapper(network).eval()

    # sanity check: wrapper must give the same result as the original network
    with torch.no_grad():
        x = torch.randn([4] + list(input_shape[1:]))
        assert torch.allclose(model(x)[1], network(x)[1], atol=1e-5), "wrapper mismatch"

    torch.onnx.export(
        model, torch.randn(input_shape), onnx_file,
        opset_version=13, dynamo=False, do_constant_folding=True,
        input_names=["input"], output_names=["features", "logits"],
    )