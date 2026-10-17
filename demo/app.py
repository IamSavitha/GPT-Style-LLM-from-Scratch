"""Gradio demo: generate TinyStories-style text with the from-scratch GPT."""
from pathlib import Path

import gradio as gr
import torch

from model import GPTModel

HERE = Path(__file__).resolve().parent
CKPT = next(p for p in [HERE / "tinygpt_checkpoint.pt", HERE.parent / "tinygpt_checkpoint.pt"] if p.exists())

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
stoi, itos = ckpt["char_to_idx"], ckpt["idx_to_char"]
model = GPTModel(vocab_size=ckpt["vocab_size"], embed_dim=256, num_heads=8, ff_dim=1024,
                 num_layers=6, block_size=ckpt["block_size"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
N_PARAMS = sum(p.numel() for p in model.parameters())


def generate(prompt, max_new_chars, temperature, top_k, seed):
    prompt = "".join(ch for ch in (prompt or "Once upon a time") if ch in stoi) or "Once upon a time"
    torch.manual_seed(int(seed))
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = model.generate(idx, int(max_new_chars), float(temperature), int(top_k) or None)
    return "".join(itos[int(i)] for i in out[0].tolist())


with gr.Blocks(title="GPT from Scratch") as demo:
    gr.Markdown(
        f"# GPT-Style LLM from Scratch\n"
        f"A {N_PARAMS / 1e6:.1f}M-parameter decoder-only Transformer written from scratch in PyTorch "
        f"(6 layers, 8 heads, character-level, context {ckpt['block_size']}), trained on TinyStories "
        f"(validation loss 0.644). [Source code](https://github.com/IamSavitha/GPT-Style-LLM-from-Scratch)"
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="Once upon a time, a little dog named")
            max_new = gr.Slider(50, 600, value=300, step=10, label="Characters to generate")
            temp = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)")
            topk = gr.Slider(0, 60, value=0, step=1, label="Top-k (0 = off)")
            seed = gr.Number(value=42, precision=0, label="Seed")
            btn = gr.Button("Generate", variant="primary")
        out = gr.Textbox(label="Output", lines=14)
    btn.click(generate, [prompt, max_new, temp, topk, seed], out)
    gr.Examples([["Once upon a time"], ["Lily found a shiny red ball"], ["The little bear was sad because"]], [prompt])

if __name__ == "__main__":
    demo.launch()
