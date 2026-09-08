# ZK Credit Scoring

ECI 2026 final project based on [ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs](https://ddkang.github.io/papers/2024/zkml-eurosys.pdf). A four-feature logistic regression predicts the Spectral liquidation-risk label. Halo2/KZG proves a single inference with private features and publicly bound model weights and output.

## Application and trust assumptions

The prover knows a preprocessed feature vector `x`. The verifier knows the approved quantized weights `w`, bias `b`, circuit configuration, and verification key. The statement is that the public integer score is the output of the configured fixed-point model on some private `x`. Weight and bias cells are exposed as public instances alongside the score, and the verifier checks them against the approved model. Five public parameters are simpler than a model-hash circuit for this tiny model.

Feature extraction, preprocessing, identity, and data authenticity are assumed correct and are **not proven**. The proof does not establish honest training, predictive accuracy, fairness, or real-world creditworthiness. Dataset inputs are already public; this is a demonstration of the protocol, not a claim that published records become secret. There is no smart contract or automatic lending decision.

## Framework

We use the [authors' implementation](https://github.com/uiuc-kang-lab/zkml) at commit `50d0cf30e2bbcb8b1f31839b775c1d018b720527`, retained as an unmodified Git submodule. TensorFlow Lite models are converted to ZKML MessagePack circuits. The proof backend is Halo2's Plonkish arithmetization with KZG on BN256, SHPLONK multi-opening, and a Blake2b Fiat–Shamir transcript. Recursion is not used.

The harness generates a local experimental KZG SRS. This is a trusted single-party setup, not the paper's multiparty ceremony. Setup and key generation are excluded from per-proof timing and reported separately.

## Dataset and model

[Spectral](https://huggingface.co/datasets/spectrallabs/credit-scoring-training-dataset) provides observations immediately before borrow events. Its challenge label includes actual liquidation and technical liquidation (health factor below 1.2), not conventional unsecured-loan default. Download revision and SHA-256 are recorded in `data/manifest.json`.

Chronological 70/15/15 boundaries are purged using a conservative 30-day gap. **The exact label horizon remains unconfirmed:** this gap must not be presented as a verified guarantee against all temporal leakage. The actual file extends through September 30, 2023, beyond the dataset card's stated August endpoint. `results/data_audit.json` records the actual observations, date boundaries, exclusions, and wallet overlap. Repeated wallets are allowed; evaluation concerns future events, not exclusively unseen wallets.

Training-only imputation, clipping, logarithms, and standardization are saved. We compare four and eight features on validation and select the smaller set within 0.01 average precision of the best. The selected inputs are `wallet_age`, `borrow_count`, `repay_count`, and `risk_factor`.

For reliable, simple fitting we use scikit-learn's LBFGS logistic regression and transfer its learned weights to the equivalent Keras `Dense(1, sigmoid)`. There is no hidden layer. The exported graph is exactly `FULLY_CONNECTED → LOGISTIC`.

Scale 128 passes validation tolerances; scale 32 does not. A vectorized fixed-point reference is checked against actual ZKML mock circuits, including extreme and near-threshold observations. Held-out test evaluation happens after model and scale selection. Complete test-set metrics use the checked reference; we do not generate a proof for every test row.

## Implementation and correctness

Python scripts handle reproducible data preparation, training, export, numerical validation and experiments. The Rust harness wraps upstream constraints, binds public model parameters, separates timings, reuses keys, saves proof artifacts and supports standalone verification without private inputs.

Tests cover chronological splits and frozen preprocessing, fixed-point rounding, acceptance of a genuine proof, and rejection of changed scores, changed model parameters and altered proof bytes. Mock tests also reject inconsistent public values. These are regression tests of concrete attacks, not a cryptographic soundness proof or an audit of upstream gadgets.

The CI workflow runs a real tiny KZG proof without downloading the dataset. CI is configured but has not run on GitHub while this repository remains local.

## Reproduction

Requirements: Git, Python 3.11, uv, Rust/rustup, and a C/C++ build toolchain. The tested machine is an Apple M2 Pro with 16 GiB RAM, macOS 14.5; proving uses four Rayon threads. Python dependencies and Rust dependencies are locked. Rust nightly is pinned in `rust-toolchain.toml`.

```sh
git clone --recurse-submodules <repository-url>
cd zk-credit-scoring
make setup build
make smoke test
make data train validate evaluate
```

Run `make calibrate optimize benchmark` for the original ZKML optimizer and the held-out comparison. Calibration is hardware-specific; regenerate it on the benchmark machine. The optimizer uses the authors' logical-plan generator and original cost formulas across all six implementations and 10–100 columns, selecting by estimated cost rather than measured proof time.

Large datasets, circuit variants, parameters and proof-run directories are ignored by Git; the small trained model, configuration, raw result JSON files and reproducible scripts are retained. Regenerate ignored outputs using the commands above.

Example standalone verification after `make smoke`:

```sh
vendor/zkml/target/release/credit-zk verify \
  artifacts/smoke/public_model.msgpack artifacts/smoke/proof 0
```

The model configuration, verification key and SRS must come from the trusted application configuration, not be accepted indiscriminately from the prover.

## Performance and scope

On 66,445 held-out test observations, FP32 ROC-AUC is 0.7731 and average precision is 0.6989. Fixed-point average precision is 0.6950, classification disagreement is 0.68%, and maximum probability deviation is 0.00801. These are predictive/numerical results, not proof timings. Full measurements are in `results/test_metrics.json`.

`results/tuning.json` retains the earlier preliminary 40-column experiments for provenance. These measurements are not used to choose either configuration in the original-optimizer experiment.

Measured on the recorded M2 Pro, ten held-out proofs per configuration:

| Metric | Fixed 40 columns | Original optimizer: 10 columns |
|---|---:|---:|
| Median proving time | 2.286 s | 0.706 s |
| Proving-time IQR | 0.028 s | 0.055 s |
| Median verification time | 17.490 ms | 6.765 ms |
| Proof size | 18,144 bytes | 5,024 bytes |
| Peak process RSS | 87.94 MiB | 37.63 MiB |

Both selected implementation 2 and `k=11` (2,048 rows). The measured speedup is **3.24×**, or **69.13% lower proving latency**, with identical integer scores. `results/benchmark.json` contains every measurement, setup/key costs and configuration fingerprints. These results are specific to this model, baseline and machine.

The primary comparison keeps the trained model, preprocessing, fixed-point scale, public/private boundary, KZG backend, observations and hardware constant. A paper-style fixed-width baseline minimizes estimated cost at 40 columns and the minimum validated `k`; the optimized configuration minimizes the same estimate over 546 candidates. All six published fully connected implementations are candidates for both.

Forty columns follows the paper's Table 10 ablation, where the width was motivated by much larger models. It is not claimed to be the best manually tuned baseline for this tiny classifier.

Small-model adapters extend the original microbenchmark domain to `k=10..14`, reject missing coefficients and prevent the arithmetic-row estimate from choosing a `k` smaller than the circuit's validated requirement. All 16 preprocessing-domain corners are checked before estimation. The upstream submodule and cost formulas remain unchanged; `build.rs` generates the adapted sources. See `docs/original-optimizer.md` for exact changes and limitations.

Final measurements use ten held-out observations per configuration in alternating blocks. We report median and IQR of proving time, verification time, proof bytes, peak process RSS, setup/key costs, calibration time and optimizer time. RSS includes setup. Feasibility checks are included in optimizer wall time, so that number is not directly comparable to the paper's optimizer latency.

Calibration took 87.9 s; optimization took 1122.1 s, including 1110.5 s of conservative feasibility checks and 6.23 s of estimator calls. This up-front cost must be amortized over repeated inferences; faster individual proofs do not imply faster one-off deployment.

The application is a small implementation of a use case already mentioned in the paper, not a novel credit-scoring proposal. A richer model, IPA comparison, authenticated data access, threshold-only disclosure, and on-chain verification are feasible extensions but out of scope for this three-day submission.

## Team contributions

To be filled by the team before submission with each member's actual contributions. AI-assisted work must be reviewed and understood by the authors.
