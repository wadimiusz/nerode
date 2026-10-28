"""The model, the training loop and the checkpoint cache.

`make_model` builds the centigpt `GPT` for an alphabet of `n_symbols` symbols
plus a BOS token. `fit` trains it as an ordinary language model on samples from
a PDFA, either on a finite dataset (the pilot setup) or on fresh samples every
batch, and caches the trained weights under a key that covers the automaton
and every training setting.

Outputs go to the directory named by the NERODE_OUT environment variable
(default: `out/` at the root of this checkout); checkpoints to its `ckpt/`.
NERODE_THREADS sets the number of CPU threads (default: at most 4).
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from centigpt.config import Config
from centigpt.model import GPT
from tqdm.auto import tqdm

from nerode.automata import PDFA, sample

# Intra-op threads for CPU runs. The model is tiny, and one thread per core
# makes every step slower, not faster: measured on a 22-core machine, a training
# step takes 528 ms with 22 threads and 20 ms with 4. NERODE_THREADS overrides.
torch.set_num_threads(
    int(os.environ.get("NERODE_THREADS", min(4, os.cpu_count() or 1)))
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_out_dir() -> Path:
    """`out/` at the root of this checkout (the package is installed in
    editable mode), so that scripts find the same checkpoint cache from any
    working directory; plain `out/` if the package is installed elsewhere."""
    root = Path(__file__).resolve().parents[2]
    return root / "out" if (root / "pyproject.toml").exists() else Path("out")


OUT_DIR = (
    Path(os.environ["NERODE_OUT"])
    if os.environ.get("NERODE_OUT")
    else default_out_dir()
)
CKPT_DIR = OUT_DIR / "ckpt"


def make_model(
    n_symbols: int, seq_len: int, d_model: int, n_layers: int, seed: int
) -> GPT:
    cfg = Config(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=2,
        vocab_size=n_symbols + 1,  # last id is BOS
        max_seq_len=seq_len + 1,
        seed=seed,
    )
    torch.manual_seed(seed)
    return GPT(cfg).to(DEVICE)


Monitor = Callable[[GPT], dict[str, str]]


def _train_loop(
    model: GPT,
    next_batch: Callable[[], torch.Tensor],
    steps: int,
    lr: float,
    desc: str | None = None,
    h_opt: float | None = None,
    monitor: Monitor | None = None,
    monitor_every: int = 5000,
) -> float:
    """The training loop shared by `train` and `train_fresh`, with a progress
    bar: smoothed training loss, its gap to the optimal loss `h_opt`, learning
    rate, and whatever `monitor(model)` reports every `monitor_every` steps.
    The bar and the monitor only read the model, so results do not change."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    bar = tqdm(
        range(steps),
        desc=desc or "training",
        unit="step",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    info: dict[str, str] = {}
    ema, n_ema = 0.0, 0  # bias-corrected moving average over ~10 readings
    loss = torch.tensor(0.0)
    for step in bar:
        xb = next_batch()
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            n_ema += 1
            ema = 0.9 * ema + 0.1 * float(loss.detach())
            smooth = ema / (1 - 0.9**n_ema)
            info["loss"] = f"{smooth:.4f}"
            if h_opt is not None:
                info["gap"] = f"{smooth - h_opt:+.4f}"
            info["lr"] = f"{sched.get_last_lr()[0]:.1e}"
            bar.set_postfix(info, refresh=False)
        if monitor is not None and (step + 1) % monitor_every == 0:
            info.update(monitor(model))
            model.train()
            bar.set_postfix(info, refresh=False)
    bar.close()
    return float(loss.detach())


def train(
    model: GPT,
    x: torch.Tensor,
    steps: int,
    lr: float,
    batch: int = 128,
    **progress,
) -> float:
    """Train on the finite dataset x (the pilot setup)."""
    gen = torch.Generator().manual_seed(0)

    def next_batch() -> torch.Tensor:
        idx = torch.randint(0, x.shape[0], (batch,), generator=gen)
        return x[idx].to(DEVICE)

    return _train_loop(model, next_batch, steps, lr, **progress)


def train_fresh(
    model: GPT,
    p: PDFA,
    steps: int,
    lr: float,
    seq_len: int,
    seed: int,
    batch: int = 128,
    **progress,
) -> float:
    """Like `train`, but every batch is a fresh sample from the automaton, so
    the model cannot overfit a finite dataset and can be trained much longer."""
    bos = p.n_symbols
    rng = np.random.default_rng(10_000 + seed)

    def next_batch() -> torch.Tensor:
        syms, _ = sample(p, batch, seq_len, rng)
        xb = np.concatenate([np.full((batch, 1), bos), syms], 1)
        return torch.tensor(xb).to(DEVICE)

    return _train_loop(model, next_batch, steps, lr, **progress)


def ckpt_path(
    model: GPT,
    p: PDFA,
    x_tr: torch.Tensor,
    *,
    steps: int,
    seed: int,
    seq_len: int,
    lr: float = 3e-3,
    batch: int = 128,
    fresh: bool = False,
) -> Path:
    """Where `fit` caches this model. The key covers the automaton itself and
    every training setting, so a changed automaton or setting never picks up a
    stale checkpoint. `x_tr` only enters the key when `fresh` is False."""
    cfg = model.cfg
    h = hashlib.sha1()
    for part in (
        p.delta.tobytes(),
        np.round(p.emit, 12).tobytes(),
        repr((steps, seed, seq_len, lr, batch, fresh)).encode(),
        repr((cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.vocab_size)).encode(),
    ):
        h.update(part)
    if not fresh:
        h.update(x_tr.numpy().tobytes())
    return CKPT_DIR / f"{h.hexdigest()[:16]}.pt"


def fit(
    model: GPT,
    p: PDFA,
    x_tr: torch.Tensor,
    *,
    steps: int,
    seed: int,
    seq_len: int,
    lr: float = 3e-3,
    batch: int = 128,
    fresh: bool = False,
    cache: bool = True,
    desc: str | None = None,
    h_opt: float | None = None,
    monitor: Monitor | None = None,
) -> GPT:
    """Train `model` in place, or load it from the checkpoint cache.

    fresh=False is the original setup (`train` on the finite dataset x_tr).
    fresh=True trains on new samples every batch (`train_fresh`)."""
    path = ckpt_path(
        model,
        p,
        x_tr,
        steps=steps,
        seed=seed,
        seq_len=seq_len,
        lr=lr,
        batch=batch,
        fresh=fresh,
    )
    if cache and path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        tqdm.write(f"  {desc or 'model'}: loaded from cache ({path.name})")
        return model
    progress = dict(desc=desc, h_opt=h_opt, monitor=monitor)
    t0 = time.time()
    if fresh:
        loss = train_fresh(model, p, steps, lr, seq_len, seed, batch, **progress)
    else:
        loss = train(model, x_tr, steps=steps, lr=lr, batch=batch, **progress)
    msg = f"  {desc or 'model'}: trained {steps} steps in {time.time() - t0:.0f}s"
    msg += f", last batch loss {loss:.4f}"
    if h_opt is not None:
        msg += f" (optimal {h_opt:.4f})"
    tqdm.write(msg)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model


@torch.no_grad()
def eval_loss(model: GPT, x: torch.Tensor) -> float:
    model.eval()
    x = x.to(DEVICE)
    logits = model(x[:, :-1])
    return float(
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
    )
