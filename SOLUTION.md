# SOLUTION
## Reproducibility

Run in Google Colab (free T4) or any machine with a CUDA GPU:

```bash
git clone <repo-url>
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```

This produces `results.json` (evaluation summary) and `predictions.csv`
(labels for `data/test.csv`).

## Determinism. 
All random sources used by the probe are seeded:
- `splitting.py` uses `random_state=42` for both `StratifiedKFold` and
  `train_test_split`.
- `probe._build_network` calls `torch.manual_seed(0)` so weight
  initialisation is identical across runs.
- `probe.fit` calls `torch.manual_seed(0)` and uses a dedicated
  `torch.Generator(seed=0)` for `torch.randperm`, so batch order and
  dropout masks are identical across runs.

The LLM forward pass runs in `eval()` mode (no dropout), so feature
extraction is deterministic too.

## Final solution

Modified files: `aggregation.py`, `probe.py`, `splitting.py`.

### `aggregation.py`
Two-layer, two-pool aggregation at the last real token position:

- last real token's hidden state at layers `-1` and `-5`;
- mean of the last 24 real tokens' hidden states at the same layers.

Concatenated, this gives `4 * 896 = 3584` features. The last-token
vector captures the model's final commitment; the tail mean adds a
local response-region signal without being washed out by the long RAG
prompt.

The optional `extract_geometric_features` returns per-layer activation
norms (averaged over real tokens) concatenated with cosine similarity
between consecutive layers at the last token (representation drift).
It is activated through `USE_GEOMETRIC`.

### `probe.py`
A small MLP: `Linear(d, 128) - ReLU - Dropout(0.3) - Linear(128, 1)`
with `StandardScaler` pre-processing. Trained for 70 epochs with Adam
(`lr=1e-3`, `weight_decay=1e-4`), mini-batch size 64.

### `splitting.py`
5-fold `StratifiedKFold`. Inside each fold, 15% of the training portion
is carved out as a validation set (used by `fit_hyperparameters` for
threshold tuning). Because every sample appears in the training portion
of at least one fold, the final probe in `solution.py` is fitted on all
labelled data, which improves predictions on `test.csv`.

### What contributed most
- Multi-layer pooling (layers `-1` and `-5`) added a clearly useful
  signal over the single-layer last-token baseline.
- 5-fold CV gave the final probe the full labelled set instead of only
  ~85%.

## Experiments and failed attempts

- Mean pooling over all real tokens (last layer) - AUROC dropped
  noticeably. The prompt is much longer than the response, so a full
  sequence mean is dominated by RAG context and drowns the response
  signal. Replaced with a tail-region mean (last 24 tokens).
- Pure last-token hidden state at 3–4 late layers did not beat the two-pool variant on this dataset.
- Single linear probe (`Linear(d, 1)`) instead of the small MLP - also
  no improvement; the MLP with dropout generalised slightly better.
- A deeper MLP (two hidden layers with two dropouts) - overfitted on
  the small dataset.
