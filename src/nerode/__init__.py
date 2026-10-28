"""Where does a transformer keep its state?

Experiments on whether a small GPT (from nanogpt), trained as a language model on
text from a probabilistic automaton, keeps the Myhill–Nerode class of a prefix in
the residual stream. Modules:

- automata:      probabilistic deterministic finite automata (PDFA), minimisation,
                 sampling; the random, shared-emission and parity constructions
- training:      the model, the training loop, the checkpoint cache
- readout:       residual streams and next-symbol distributions of a model
- stats:         rank correlations and distance matrices
- geometry:      version 1: the residual vector at the last position of a prefix
- continuations: version 2: prefixes compared through their continuations
- probes:        probes for the hidden parity bit

The command-line scripts in scripts/ are thin wrappers around these modules.
"""
