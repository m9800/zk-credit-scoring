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
- Implemented reproducible bounded empirical tuning and final paired benchmark scripts.
- Added English documentation, locked dependencies, seven passing local tests and a CI workflow.

## Pending decisions and work

1. Confirm whether to use the smaller empirical configuration search or calibrate the original paper's cost estimator. The current 40-column measurements are tuning observations, not final benchmark results or a claimed speedup.
2. Execute the selected two-configuration held-out benchmark and add its table to the README.
3. Confirm the Spectral target horizon from an authoritative source. The 30-day gap is explicitly provisional; do not claim verified leakage-free labels.
4. Team review of code, assumptions and results; add actual member contributions.
5. Publish the repository only at the user's requested final stage. CI has not run remotely yet.

## Differences from the teammate's proposal

- Compatibility smoke test comes first.
- Logistic regression is fitted by scikit-learn LBFGS and copied into an equivalent Keras model, avoiding neural-network optimizer tuning.
- Public weights and bias directly bind the approved model, avoiding a hash gadget for five numbers. Features remain private, the integer score is public.
- Scale is selected on validation, never on test.
- Baseline selection uses only validation observations; final benchmark uses separate test observations.
- The proposed simplified search is explicitly described as empirical tuning, not the full ZKML optimizer.
- KZG SRS generation is local and experimental, not a multiparty ceremony.
- Claim future-event prediction for the published liquidation-risk label; do not claim generalized unsecured creditworthiness or unseen-wallet generalization.

## Next benchmark command, if empirical tuning is selected

```sh
make benchmark
```

It evaluates 30 configurations (five widths and six implementations), with minimum valid row counts for the preprocessed domain. Each candidate gets one warmup and three validation proofs. Two selected configurations then process the same ten seeded test observations in alternating blocks. Setup, key generation and file I/O are outside the proof/verification timers. All trial data are retained; unsuccessful capacity checks are logged.

Direct empirical search is a deliberate scope reduction: the experiment answers whether tuning helps this tiny application, not whether the original estimator predicts hardware costs accurately.
