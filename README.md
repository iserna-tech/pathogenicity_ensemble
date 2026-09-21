# Pathogenicity Ensemble

Predicting whether a protein-coding variant is pathogenic by combining **protein
language model embeddings**, **predicted 3D structure**, and a **graph neural
network over the wild-type / mutant structural difference**.

## Motivation

Sequence-only variant classifiers ignore what a mutation actually does to the
fold. This project folds both the wild-type (WT) and mutated (MUT) protein with
OpenFold3, aligns the two structures, and feeds the *structural delta* — per
residue displacement, contact-map changes, coordinate deltas — into a GNN
alongside ESM2 sequence embeddings. The three views are then combined by a
stacked ensemble.

## What was done

The notebook runs end to end in four stages.

### 1. Data preparation

* Input is `protein_mutation_dataset_APC.csv` — ClinVar-derived APC gene
  variants with `normal_sequence`, `mutated_sequence`, HGVS notation, allele /
  variation IDs, and a `label` (`pathogenic` / `benign`). The run captured in
  the notebook used **51 samples**.
* Labels are binarized: `pathogenic` / `likely pathogenic` → 1, everything
  else → 0.

### 2. Structure prediction with OpenFold3

* APC is ~2,843 residues, far past what fits on a single GPU for folding, so
  each variant is reduced to a **±250 residue window centered on the first
  differing residue** (501 residues total). The same window is cut from both
  the WT and MUT sequence.
* Each window is written as an OpenFold3 query JSON and folded with
  `run_openfold predict`, producing `.cif` models (102 queries = 51 samples ×
  2 variants).
* `cueq_gpu_runner.yml` holds a low-memory inference profile (chunk size 1,
  LMA attention, MSA-module and confidence-head offloading, per-sample token /
  atom cutoffs) so folding fits on a single A100 40 GB. Runs are idempotent —
  a sample with existing CIFs is skipped — and results are zipped for reuse.

### 3. Structural comparison

* CA coordinates are extracted from each CIF with `gemmi`.
* The MUT structure is superimposed on the WT with a **Kabsch alignment**
  (SVD-based, with a reflection guard).
* Per-residue displacement, contact maps at an 8 Å cutoff, and radius of
  gyration are computed from the aligned pair; coordinates are exported to CSV
  and WT vs. mutation-replaced geometry is rendered side by side in `py3Dmol`.

### 4. Feature building, models, and the ensemble

Three complementary base learners are trained on every fold:

| Branch | Input | Model |
| --- | --- | --- |
| `ESM2_MLP` | Mean ESM2 (`esm2_t6_8M_UR50D`) embeddings of the full WT and MUT sequences, their difference and absolute difference → standardized → PCA (32 components) | 3-layer MLP |
| `STRUCT_MLP` | 16 hand-built structural descriptors: RMSD, mean/std/max/p90 displacement, displacement at the mutation site, local (±10 residue) displacement, WT/MUT/Δ radius of gyration, contact-change count and fraction, local contact change | small MLP with label smoothing and gradient clipping |
| `GNN` | One graph per variant: nodes = residues, node features = per-residue ESM2 delta ‖ WT amino-acid one-hot ‖ displacement ‖ coordinate delta ‖ normalized distance from the mutation ‖ mutation mask ‖ near-mutation mask | 2-layer `GINEConv` |

The graph edges are **mutation-aware**: backbone edges plus any pair within
8 Å in *either* structure, with edge attributes carrying WT distance, MUT
distance, their difference, and a flag for contacts that appear or disappear.
Pooling concatenates mean, max, near-mutation-masked and mutation-masked
readouts plus four raw scalar summaries before the classifier head.

**Stacking.** A 10-fold outer `StratifiedKFold` wraps a 3-fold inner CV. The
inner folds generate out-of-fold base predictions that train a balanced
logistic-regression meta-learner; the outer fold then scores it. A fixed-weight
blend (0.80 ESM2 / 0.10 GNN / 0.10 structure) is evaluated as a baseline
alongside it. Accuracy, ROC-AUC, average precision, log loss and F1 are
reported per fold and pooled over the out-of-fold predictions.

## Main features

* **Windowed folding** — makes large proteins tractable by folding only the
  neighborhood the mutation can plausibly affect.
* **Paired WT/MUT prediction with Kabsch superposition** — the signal is the
  *change* in structure, not either structure alone.
* **Mutation-aware graph construction** — edges encode contact gain/loss, nodes
  encode distance from the mutation site.
* **ESM2 sequence deltas** at both the whole-protein and per-residue level.
* **Nested-CV stacked ensemble** so the meta-learner never sees predictions
  from models trained on its own validation data.
* **Caching throughout** — folded CIFs, built graphs (`gnn_cache/*.pt`) and ESM
  features (`*.npz`) are all reused across runs.

## Requirements

Built for Google Colab with an **A100 40 GB** GPU and Google Drive mounted.

```
openfold3[cuequivariance]  nvidia-cutlass  fair-esm  torch  torch-geometric
gemmi  py3Dmol  biopython  pandas  numpy  scikit-learn
```

Run `setup_openfold` once to download the OpenFold3 checkpoint
(`openbind-2025-06-30-174k`).

## Running it

1. Place `protein_mutation_dataset_APC.csv` next to the notebook.
2. Run cells 1–6 to install dependencies, set CUDA/CUTLASS environment
   variables, and write the runner YAML.
3. Run the single-sample cells (7–10) to sanity-check one variant end to end
   and inspect the WT/MUT overlay.
4. Run cells 11–13 to generate all 102 window queries, fold them, and zip the
   results into `all_openfold_predictions.zip`.
5. Run the final cell to build features and train/evaluate the ensemble.

## Status and caveats

* **No evaluation results are saved in the notebook.** The final ensemble cell
  has no stored output, so this README deliberately reports no accuracy or AUC
  numbers — run it yourself to get them.
* `protein_mutation_dataset_APC.csv` is **not checked into this repository**;
  the notebook expects it in the working directory.
* The dataset is small (51 samples, one gene) and window-based folding assumes
  the first differing residue marks the region of interest — which is a weak
  assumption for frameshift and nonsense variants, where everything downstream
  changes.
* Deduplication keeps the lexicographically first CIF per sample/variant rather
  than the highest-confidence model; OpenFold3 confidence scores are not yet
  used as features.

## License

MIT — see [LICENSE](LICENSE).
