# Implementation status

The submission is due September 10, 2026. Keep one trained logistic regression and KZG only.

## Completed locally

- Initialized the local Git repository; nothing has been published.
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
- Added English documentation, locked dependencies, twelve passing local tests and a CI workflow.

## Pending decisions and work

1. Confirm the Spectral target horizon from an authoritative source. The 30-day gap is explicitly provisional; do not claim verified leakage-free labels.
2. Team review of code, assumptions and results; add actual member contributions.
3. Publish the repository only at the user's requested final stage. CI has not run remotely yet.

## Differences from the teammate's proposal

- Compatibility smoke test comes first.
- Logistic regression is fitted by scikit-learn LBFGS and copied into an equivalent Keras model, avoiding neural-network optimizer tuning.
- Public weights and bias directly bind the approved model, avoiding a hash gadget for five numbers. Features remain private, the integer score is public.
- Scale is selected on validation, never on test.
- Baseline selection uses only validation observations; final benchmark uses separate test observations.
- Following the team's decision, the original calibrated cost estimator replaces the proposed empirical search. Earlier 40-column measurements are retained only as preliminary data, not used for selection.
- KZG SRS generation is local and experimental, not a multiparty ceremony.
- Claim future-event prediction for the published liquidation-risk label; do not claim generalized unsecured creditworthiness or unseen-wallet generalization.

## Original-optimizer benchmark commands

```sh
make calibrate optimize benchmark
```

It evaluates 546 configurations (every width from 10 through 100 and six original implementations). Candidate selection uses the original calibrated cost formula, not real proof timings. Two selected configurations then process the same ten seeded test observations in alternating blocks. Setup, key generation and file I/O are outside the proof/verification timers. Calibration, all candidate estimates and unsuccessful capacity checks are retained.

See `original-optimizer.md` for the small-model compatibility adapters. The cost formula is reused, while operational feasibility is validated before ranking.
