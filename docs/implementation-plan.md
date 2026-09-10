# Implementation status

This project evaluates one trained logistic regression using KZG proofs. The submission deadline is September 10, 2026.

## Completed

- Published the repository at https://github.com/m9800/zk-credit-scoring.
- Pinned the original ZKML source as an unmodified submodule.
- Built upstream binaries and a separate application proof harness.
- Downloaded and hashed the pinned Spectral Parquet file (442,961 rows).
- Created chronological train/validation/test partitions with a provisional 30-day gap.
- Trained four- and eight-feature logistic candidates; selected four on validation.
- Exported the chosen model to Keras and TFLite with exactly two operators.
- Selected fixed-point scale 128 using validation data only.
- Cross-checked the fixed-point reference on 16 actual validation circuits.
- Evaluated the frozen model on all 66,445 held-out test observations.
- Generated a real KZG smoke proof and verified it independently without its witness.
- Tested rejection of modified scores, public model weights and proof bytes.
- Checked all six fully connected implementations at 40 columns, including feature-domain corners.
- Integrated the original logical-plan generator, hardware calibration and original cost model with explicit small-circuit adapters.
- Calibrated all five KZG primitives on the current machine; retained raw Criterion samples.
- Implemented the full 10–100-column / six-implementation search and final paired benchmark scripts.
- Completed all 546 candidates: estimator selected implementation 2 at 10 columns, k=11; the 40-column baseline independently selected implementation 2, k=11.
- Completed ten held-out proofs per configuration: median proving time 2.286 s → 0.706 s (3.24× speedup), with identical scores and all tampering checks rejected.
- Added English documentation, locked dependencies, local tests and a CI workflow. The [GitHub CI run for commit `6f9aa6f`](https://github.com/m9800/zk-credit-scoring/actions/runs/34416952809) passed.

## Remaining work

1. Confirm the Spectral target horizon from an authoritative source. Until then, the 30-day gap does not guarantee leakage-free labels.
2. Team review of code, assumptions and results; add actual member contributions.
3. Reproduce the full experiment on a second machine. CI checks the small proof and tests, not the complete dataset and optimizer benchmark.

## Design decisions

- A small smoke test checks framework compatibility before the full experiment.
- Logistic regression is fitted by scikit-learn LBFGS and copied into an equivalent Keras model, avoiding neural-network optimizer tuning.
- Public weights and bias directly bind the approved model, avoiding a hash gadget for five numbers. Features remain private, the integer score is public.
- Scale is selected on validation, never on test.
- Both circuit configurations are selected by estimated cost after checking the 16 corners of the feature domain defined by training-only preprocessing. Neither validation observations nor test observations are used to rank circuit candidates. The final benchmark uses held-out test observations.
- The original calibrated cost estimator selects the circuits. Earlier 40-column proof measurements are retained as preliminary data, not used for selection.
- KZG SRS generation is local and experimental, not a multiparty ceremony.
- Evaluation measures future-event prediction for the published liquidation-risk label, not unsecured creditworthiness or generalization to unseen wallets.
- Published JSON reports use repository-relative paths; local working files can retain absolute paths. Existing reports were updated only to remove the machine-specific path prefix, without changing measurements or artifact hashes.

## Original-optimizer benchmark commands

```sh
make calibrate optimize benchmark
```

It evaluates 546 configurations (every width from 10 through 100 and six original implementations). Candidate selection uses the original calibrated cost formula, not real proof timings. Two selected configurations then process the same ten seeded test observations in alternating blocks. Setup, key generation and file I/O are outside the proof/verification timers. Calibration, all candidate estimates and unsuccessful capacity checks are retained.

See `original-optimizer.md` for the small-model compatibility adapters. The cost formula is reused, while operational feasibility is validated before ranking.
