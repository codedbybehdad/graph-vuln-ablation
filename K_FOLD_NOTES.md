# 5-fold cross-validation update

The training pipeline now uses **stratified 5-fold cross-validation by default**.

## What changed

- `scripts/training/train_ggnn.py`
  - Replaced the single fixed 75/25 `train_test_split` with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
  - The exact same fold assignments are reused for all seven edge configurations.
  - Each fold trains on ~80% of the graphs and validates on ~20%.
  - Each graph is used as validation exactly once.
  - A separate best checkpoint, metrics JSON, and training curve are saved for every fold/configuration.
  - A summary JSON reports the mean and sample standard deviation across the five folds for Accuracy, Precision, Recall, F1, and AUC.
  - The exact validation-fold assignment is saved so the experiment is auditable and reproducible.

- `main.py`
  - The full seven-configuration experiment is now sent to `train_ggnn.py --edges all --folds 5` in one call, so one shared set of fold partitions is used across the ablation study.

- `README.md`
  - Added the 5-fold commands and output description.

## Commands

Run all seven ablations on QEMU:

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges all --folds 5
```

Run all seven ablations on FFmpeg:

```bash
python scripts/training/train_ggnn.py --dataset ffmpeg --edges all --folds 5
```

Run only the complete graph configuration:

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges ast+cfg+pdg --folds 5
```

Run the full QEMU + FFmpeg pipeline:

```bash
python main.py --full-experiment
```

## Interpretation

5-fold CV is an evaluation protocol. It reduces dependence on one particular random train/validation split, but it should not be used to force `AST + CFG + PDG` to be the highest-performing configuration. The result should be whatever the five held-out folds measure.

Also note that the thesis implementation is not a paper-identical Devign reproduction: the current project uses AST/CFG/PDG edge categories and a different GGNN/classifier structure, whereas the original Devign paper uses AST/CFG/NCS plus three data-flow relations in its composite representation. The original paper also reports a random 75% train / 25% validation split rather than 5-fold CV.
