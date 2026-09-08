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

`make benchmark` runs bounded empirical configuration tuning and then the held-out benchmark. This is explicitly **not** the original paper's calibrated cost-estimator optimizer. The benchmark scope is being confirmed with the team before final measurements.

Large datasets, circuit variants, parameters and proof-run directories are ignored by Git; the small trained model, configuration, raw result JSON files and reproducible scripts are retained. Regenerate ignored outputs using the commands above.

Example standalone verification after `make smoke`:

```sh
vendor/zkml/target/release/credit-zk verify \
  artifacts/smoke/public_model.msgpack artifacts/smoke/proof 0
```

The model configuration, verification key and SRS must come from the trusted application configuration, not be accepted indiscriminately from the prover.

## Performance and scope

On 66,445 held-out test observations, FP32 ROC-AUC is 0.7731 and average precision is 0.6989. Fixed-point average precision is 0.6950, classification disagreement is 0.68%, and maximum probability deviation is 0.00801. These are predictive/numerical results, not proof timings. Full measurements are in `results/test_metrics.json`.

The six fixed-width tuning candidates all pass at `k=11`; three validation proofs per candidate give median proving times around 2.29–2.40 seconds on the recorded machine. These preliminary tuning measurements are **not** the final held-out comparison. See `results/tuning.json` and `docs/implementation-plan.md` for status.

The primary comparison keeps the trained model, preprocessing, fixed-point scale, public/private boundary, KZG backend, observations and hardware constant. A paper-style fixed-width baseline uses 40 columns and the minimum valid `k`; the alternative explores a bounded set of column counts. All six published fully connected implementations are candidates for both, subject to validity and matching numerical outputs.

Tuning uses validation observations; final measurements use held-out observations. We report median and IQR of proving time, verification time, proof bytes, peak process RSS, setup/key costs, and tuning time. RSS is a process-wide peak including setup, not an isolated prover allocation measurement. No speedup is assumed in advance.

The application is a small implementation of a use case already mentioned in the paper, not a novel credit-scoring proposal. A richer model, IPA comparison, authenticated data access, threshold-only disclosure, and on-chain verification are feasible extensions but out of scope for this three-day submission.

## Team contributions

To be filled by the team before submission with each member's actual contributions. AI-assisted work must be reviewed and understood by the authors.
