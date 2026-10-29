# Where does a transformer keep its state?

Experiments on whether a small transformer, trained as an ordinary language
model on text from a probabilistic automaton, keeps the Myhill–Nerode class of
a prefix in the residual stream. An interactive write-up of the results, with a
demo: [wadimiusz.me/nerode](https://wadimiusz.me/nerode/).

The model is the `GPT` from [centigpt](https://github.com/wadimiusz/centigpt),
a GPT-style transformer written from scratch; this repository depends on it as
a package, pinned to a commit.

The question, the hypothesis and the idea of the pilot are mine; the
Myhill–Nerode framing came out of discussions with Claude, which also wrote the
experiment code, proposed some of the controls, and worked through the
interpretation with me.

## The question

Two prefixes are Myhill–Nerode equivalent when every continuation gives them
the same future. The equivalence classes are the states of the smallest
automaton for a language, and the definition does not mention any
architecture. The long-term goal is to read these classes off a trained
transformer from its weights. These experiments check a first, simpler
question: does a trained transformer keep the class of a prefix in the residual
stream at the prefix's last position?

## Setup

All scripts train the `GPT` from centigpt (2 layers, `d_model` 64) as an
ordinary language model on text sampled from a small probabilistic automaton,
where the classes are known exactly:

- **plain**: a random automaton in which every state has its own next-symbol
  distribution;
- **shared emissions**: states come in pairs that predict the next symbol
  identically and differ only further ahead;
- **parity**: three groups of states, each with a hidden bit. `0` flips the
  bit, `1` keeps it, `2` reads it (the next group depends on it) and resets it.
  The last symbol usually does not reveal the bit.

## Running the experiments

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/). `make install`
(or `uv sync`) creates the environment, including the pinned centigpt.

Run from the repository root. Everything works on a CPU; a GPU is used when
available. Outputs go to `out/` (override with the `NERODE_OUT` environment
variable); trained models are cached in `out/ckpt/`, keyed by the automaton
and every training setting, so a second run only repeats the analysis. On a
CPU the model is small enough that many threads slow it down, so at most 4 are
used; `NERODE_THREADS` overrides that.

| Script | What it does | CPU time |
| --- | --- | --- |
| `scripts/nerode_geometry.py` | Probes the residual stream at each position for the class; compares with an untrained model and with what position and last symbol alone predict | ~5 min |
| `scripts/where_errors.py` | Measures where the trained model's predictions diverge from the automaton | ~2 min |
| `scripts/nerode_continuations.py` | Compares two prefixes through every common continuation, against the model's own behaviour; `--w-symbols 0 1` keeps the parity bit hidden | ~10 min |
| `scripts/demo_data.py` | Exports the data for the interactive demo | ~4 min |
| `scripts/bit_probe.py` | Linear and MLP probes for the hidden bit on the cached parity models, by symbols since the last `2`; does the model apply the bit after a `2`? | needs the cached models |
| `scripts/bit_probe_by_gap.py` | The same probes against window baselines, by exact gap | needs the cached models |
| `scripts/bit_probe_long_gaps.py` | Probes trained only on long-gap positions | needs the cached models |
| `scripts/extract_automaton.py` | Extracts each parity model's automaton from its behaviour (distributions over continuations, clustered at radius eps) and compares it with the true one: states, purity, KL on held-out sequences | needs the cached models |
| `scripts/train_sweep.py` | How many steps a parity model needs to reach the optimal loss | ~1 min per 1000 steps |

```bash
uv run python scripts/nerode_continuations.py --quick       # 30-second smoke run
uv run --extra plots python scripts/nerode_geometry.py      # with PCA plots
```

### Training longer, and checking that the model learned the automaton

The pilot models were trained for 3000 steps on a fixed set of 5000 sequences,
which left them short of the optimal loss. Three flags change that, and the
defaults reproduce the pilot exactly:

- `--fresh` draws a new batch from the automaton at every step, so training
  can run much longer without overfitting a finite dataset;
- `--min-sep 0.1` (parity automata) redraws emissions until every two groups
  differ by at least that much KL; otherwise two groups can predict the next
  symbol almost identically, and the hidden bit then barely matters;
- `--kinds parity_hand` (and `demo_data.py --kind parity_hand`) uses one parity
  automaton chosen by hand, with clearly different groups.

For parity automata the output also reports how well the trained model tracks
the hidden bit, by prefix length. The well-trained models behind the findings
below, and the probes that need them, come from:

```bash
uv run python scripts/nerode_continuations.py --kinds parity_hand parity \
    --seeds 0 1 2 3 4 --fresh --steps 100000 --min-sep 0.1 --out out/continuations_fixed.txt
uv run python scripts/nerode_geometry.py --fresh --steps 100000
uv run python scripts/demo_data.py --kind parity_hand --fresh --steps 100000
uv run python scripts/bit_probe.py
```

## Results

The small result files are committed under `results/`; checkpoints and plots
are not.

| File | Produced by |
| --- | --- |
| `results/geometry.txt` | `nerode_geometry.py` with the pilot defaults (1500 steps on 5000 fixed sequences) |
| `results/geometry_fixed.txt` | `nerode_geometry.py --fresh --steps 100000` (findings 1 and 2) |
| `results/continuations.txt` | `nerode_continuations.py` with the pilot defaults |
| `results/continuations_parity_w01.txt` | `nerode_continuations.py --kinds parity --w-symbols 0 1` |
| `results/continuations_fixed.txt` | `nerode_continuations.py --kinds parity_hand parity --seeds 0 1 2 3 4 --fresh --steps 100000 --min-sep 0.1` (finding 3, first table) |
| `results/bit_probe_concat.txt`, `.json` | `bit_probe.py` on those ten models (finding 3, probes) |
| `results/demo_data.json` | `demo_data.py --kind parity_hand --fresh --steps 100000` (the demo) |
| `results/sweep/` | `train_sweep.py --fresh-data --steps 10000 30000 100000`, one seed per file; the `.jsonl` is the streamed log of the same runs |

## What I found

The models are small (2 layers, width 64), each trained for 100,000 steps on
fresh samples: one model per automaton for findings 1 and 2, five automata of
each kind for finding 3.

### 1. Classes that change the next symbol are visible at the current position

On random automata where every state predicts the next symbol differently, a
linear probe recovers the class from the residual stream far better than
position and last symbol alone can.

| Automaton | Probe on trained model | Position + last symbol | Probe on untrained model |
| --- | --- | --- | --- |
| 3 classes | 1.00 | 0.80 | 0.78 |
| 6 classes | 0.98 | 0.49 | 0.60 |
| 8 classes | 0.97 | 0.52 | 0.52 |

Residual stream after the last layer, one model per automaton. This is partly
forced: the last layer has to tell these classes apart to predict the next
symbol at all.

### 2. Differences that matter only later are mostly not stored there

When two classes predict the same next symbol and differ only further ahead, a
probe after the last layer separates them with accuracy 0.87–0.92, against
0.84–0.85 from position and last symbol and 0.86–0.88 from an untrained model.
The model does not need that difference yet, and mostly does not keep it in
that vector. After the first layer the probe does a little better, 0.95, which
fits the short window of recent symbols described in finding 3.

### 3. Over long gaps the hidden bit is mostly recomputed when it is needed

For the automaton in the demo, and for five random ones, I compared prefixes
through their continuations: the residual stream after the second layer at the
end of *u* followed by each short continuation *w*. The score is 0 when the
hidden bit is invisible and 1 when it separates prefixes as much as a different
group does. Five models per row; the range is over models.

| Automata | At the end of *u* | After *w* without a `2` (bit not needed yet) | After any *w* |
| --- | --- | --- | --- |
| Hand-chosen (demo) | 0.00–0.01 | 0.00–0.01 | 0.66–0.68 |
| Random, groups forced apart | 0.00–0.02 | 0.00–0.03 | 0.67–1.20 |

Every model reaches the optimal loss to within 0.006 nats and tracks the hidden
bit on every prefix of up to 7 symbols. Scores above 1 mean that after a `2`
the bit separates prefixes even more than a different group does.

This score compares whole vectors, so it could miss a bit kept in a small or
nonlinear part of the vector. I therefore also trained probes, linear and a
small MLP, to read the bit from the residual stream at the current position. In
case the bit is spread across layers rather than held in one, as
[Shai et al. (2024)](https://arxiv.org/abs/2405.15943) found in a similar
case, I also probed the token embedding and both blocks concatenated. The last
row is a baseline: it guesses the bit from the group, the last six symbols and
the gap, so it catches everything a short window of recent symbols can explain.

| Symbols since the last `2` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11–13 | 14–23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP probe, last layer | 0.98 | 0.97 | 0.97 | 0.95 | 0.90 | 0.83 | 0.75 | 0.65 | 0.64 | 0.68 | 0.61 | 0.57 |
| MLP probe, all layers | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.90 | 0.82 | 0.72 | 0.65 | 0.63 | 0.58 | 0.53 |
| Same probe, untrained model | 0.94 | 0.76 | 0.70 | 0.68 | 0.65 | 0.59 | 0.56 | 0.56 | 0.53 | 0.50 | 0.46 | 0.49 |
| Group + last six symbols | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.69 | 0.65 | 0.64 | 0.52 | 0.52 | 0.43 |

Accuracy of predicting the bit on held-out sequences, averaged over the ten
models; positions right after a `2` are left out, since the bit is always 0
there. Long gaps are rare (about 300 test positions at 14–23 across all
models), so the last columns are noisy.

A probe trained on all positions sees few long gaps and may simply not learn
them. Trained only on positions at least 11 symbols after the last `2`, the MLP
reads the bit with accuracy 0.62 on average, against 0.53–0.58 for the
untrained twin and 0.50 for the majority class. The average hides a split: in
two of the ten models the bit is readable at 0.81–0.84, in most of the others
barely.

So the vector at the current position carries a window of recent symbols, five
or six long when the layers are taken together, which is enough to recover the
bit when a `2` was a few symbols back. Beyond that window most models keep at
most a weak trace of the bit, and a few keep a partial running parity. Yet all
of them apply the bit correctly after long gaps: right after a `2` the model
picks the right next group 99.7% of the time when the previous `2` was up to
six symbols back, and 96% of the time when it was ten or more. In most models,
then, the bit is largely computed from the cache when the `2` arrives, and how
much of it a model carries forward varies from seed to seed.

## What follows

The state of a transformer is spread over its whole key-value cache. Each
position keeps a short window of recent symbols, and older information that
matters only later is mostly computed from the cache when it is needed. Nothing
in the cache is ever overwritten: the prefix is all there. What compresses it
is reading. Later positions see the past only through attention, so two
prefixes are equivalent for the model when no continuation can read their
caches differently.

That gives the goal a precise form. Treat the transformer as an automaton whose
state is its cache, and compute its minimal automaton, the cache states modulo
this indistinguishability, from the weights. The way there is to find what
attention can read from the past: summaries of the prefix such as counts,
parities, or the position of the last occurrence of a symbol, and show that the
Myhill–Nerode classes are their level sets. Transformers compiled from known
automata, where these summaries are put in by hand, are the test bed.

Related work and the change log are on the
[write-up page](https://wadimiusz.me/nerode/).

## Layout

| Path | Contents |
| --- | --- |
| `src/nerode/automata.py` | `PDFA`, random and shared-emission automata, parity automata, minimisation, sampling |
| `src/nerode/training.py` | The model (`make_model`), the training loop, `fit` with the checkpoint cache |
| `src/nerode/readout.py` | Residual streams, next-symbol distributions, hidden-bit accuracy of a model |
| `src/nerode/stats.py` | Rank correlations, cosine and total-variation distance matrices |
| `src/nerode/geometry.py` | Version 1: the residual vector at the last position (`run_one`, the metrics, PCA plots) |
| `src/nerode/continuations.py` | Version 2: prefixes compared through continuations (`run_one`, `hidden_score`) |
| `src/nerode/probes.py` | Probes for the hidden bit: features, linear and MLP probes, window baselines |
| `scripts/` | The command-line entry points listed above |
| `tests/` | Smoke tests, including the `--quick` continuations run |
| `results/` | Committed result files, see above |

## Checks

`make check` runs the linter, the formatter check, the type checker and the
tests, without modifying files; `make fix` applies ruff's autofixes and
formatting. The settings are the same as in centigpt and live in
`pyproject.toml`.

## License

MIT
