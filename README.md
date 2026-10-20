# Experiments: where does a transformer keep its state?

An interactive write-up of these results, with a demo:
[wadimiusz.me/nerode](https://wadimiusz.me/nerode/).

## The question

Two prefixes are Myhill–Nerode equivalent when every continuation gives them
the same future. The equivalence classes are the states of the smallest
automaton for a language, and the definition does not mention any
architecture. The long-term goal is to read these classes off a trained
transformer from its weights. These experiments check a first, simpler
question: does a trained transformer keep the class of a prefix in the residual
stream at the prefix's last position?

## Setup

All scripts train the `GPT` from this package (2 layers, `d_model` 64) as an
ordinary language model on text sampled from a small probabilistic automaton,
where the classes are known exactly:

- **plain**: a random automaton in which every state has its own next-symbol
  distribution;
- **shared emissions**: states come in pairs that predict the next symbol
  identically and differ only further ahead;
- **parity**: three groups of states, each with a hidden bit. `0` flips the
  bit, `1` keeps it, `2` reads it (the next group depends on it) and resets it.
  The last symbol usually does not reveal the bit.

## Scripts

Run from the repository root. Everything works on a CPU; a GPU is used when
available. Outputs go to `experiments/nerode_out/`.

| Script | What it does | CPU time |
| --- | --- | --- |
| `nerode_geometry.py` | Probes the residual stream at each position for the class; compares with an untrained model and with what position and last symbol alone predict | ~5 min |
| `where_errors.py` | Measures where the trained model's predictions diverge from the automaton | ~2 min |
| `nerode_continuations.py` | Compares two prefixes through every common continuation, against the model's own behaviour; `--w-symbols 0 1` keeps the parity bit hidden | ~10 min |
| `demo_data.py` | Exports the data for the interactive demo | ~4 min |

```bash
uv run python experiments/nerode_continuations.py --quick   # 30-second smoke run
uv run --with matplotlib python experiments/nerode_geometry.py   # with plots
```

## What we found

1. Classes that change the next symbol are linearly readable at the current
   position, well above both baselines. This is partly forced: the last layer
   has to separate them to predict the next symbol.
2. Classes that predict the same next symbol and differ only later are barely
   more readable than from position and last symbol alone.
3. On parity automata, the hidden bit becomes visible in the geometry when a
   `2` makes it matter for the next symbol, and in two of three seeds not
   before.

So the state is spread over the key-value cache, and the part needed later is
computed when it is needed. The next step is to describe the model's classes as
what attention can read from the cache, starting from transformers compiled
from known automata.

These are pilot results: one small model per seed, trained on a CPU, with
linear (cosine and probe) measures of geometry.
