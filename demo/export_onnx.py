"""Export the trained checkpoint to ONNX + vocab for the in-browser demo (demo/web/).

    pip install torch onnx
    python demo/export_onnx.py          # writes demo/web/model.onnx and demo/web/vocab.json

Run by .github/workflows/pages.yml on every push, so the browser demo always matches the checkpoint.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import GPTModel  # noqa: E402

ckpt = torch.load(HERE.parent / "tinygpt_checkpoint.pt", map_location="cpu", weights_only=False)
model = GPTModel(vocab_size=ckpt["vocab_size"], embed_dim=256, num_heads=8, ff_dim=1024,
                 num_layers=6, block_size=ckpt["block_size"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

out = HERE / "web"
out.mkdir(exist_ok=True)
dummy = torch.zeros(1, 16, dtype=torch.long)
torch.onnx.export(
    model, (dummy,), out / "model.onnx",
    input_names=["input_ids"], output_names=["logits"],
    dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
    opset_version=17, do_constant_folding=True,
)

itos = ckpt["idx_to_char"]
vocab = [itos[i] for i in range(ckpt["vocab_size"])]
(out / "vocab.json").write_text(json.dumps({
    "itos": vocab,
    "block_size": ckpt["block_size"],
    "params": sum(p.numel() for p in model.parameters()),
}))

# Sanity check: ONNX logits match PyTorch.
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

ids = torch.randint(0, ckpt["vocab_size"], (1, 40))
ref = model(ids).detach().numpy()
got = ort.InferenceSession(str(out / "model.onnx")).run(None, {"input_ids": ids.numpy()})[0]
err = float(np.abs(ref - got).max())
print(f"exported {out / 'model.onnx'}  max |torch - onnx| = {err:.2e}")
assert err < 1e-3, "ONNX output does not match PyTorch"
