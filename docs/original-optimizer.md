# Original ZKML optimizer integration

## Scope

The experiment uses the authors' released logical-plan generator and hardware-calibrated cost model. It does not choose the best configuration by measuring candidate proof times. The fixed model, scale 128, KZG backend, public weights and private feature vector are identical on both sides of the comparison.

The public `find_optimal.py` orchestrator invokes `python/create_logical.py`, enumerates implementations and column counts, calls `estimate_cost`, and minimizes its output. Our `scripts/optimize.py` performs these same stages with safe argument arrays and sequential processes, preserving reproducible logs. It replaces shell orchestration, not the mathematical cost model. The public release should not be confused with an exact reproduction of every internal optimizer experiment in the paper.

## Reused upstream code

- `python/create_logical.py` generates all six implementations for the fully connected layer; Logistic has one implementation.
- `src/bin/estimate_cost.rs` counts circuit operations, FFTs, MSMs, permutation work and arithmetic, then combines them with measured coefficients.
- The five `benches/benchmark_kzg_*.rs` microbenchmarks supply these coefficients through Criterion.

The submodule remains unmodified. `build.rs` reads its pinned sources into Cargo's generated-source directory and applies these explicit adapters:

1. Move the unstable Rust feature attribute to the including wrapper.
2. Set the microbenchmark range to `k=10..14`, covering the small circuits and extended FFT domains. The original range is `13..19`.
3. Use `max(original_arithmetic_row_estimate_k, validated_circuit_k)` before evaluating cost. For tiny circuits, the original arithmetic-row count alone omits important numerical/lookup feasibility limits. This prevents ranking unrealizable configurations.
4. Fail on missing calibration entries instead of returning zero cost.

The operation-count and cost-estimation functions are otherwise reused verbatim. There are no changes to inference constraints or to the model's scale. `build.rs` asserts that each replacement occurs exactly once, preventing silent application to a changed source.

## Feasibility and statement

The trained preprocessing bounds define a four-dimensional box. The search begins with a numeric-range lower bound on `k`, then checks all 16 corners with actual MockProver constraints for each candidate. It increases `k` until these checks pass. This is a conservative operational feasibility check, not a proof that finite testing audits every upstream gadget.

Every candidate must expose exactly the approved quantized weights/bias and the expected integer score. Valid mock executions and rejection of altered public scores/weights are checked before estimation. The same model identity is verified again for the selected real proofs.

## Selection and measurement

The search space is six original logical implementations × 91 column counts (10 through 100 inclusive). The baseline minimizes estimated proving cost among valid 40-column candidates. The optimized configuration minimizes the same estimate across the complete search space. Ties are broken deterministically by implementation index and column count.

The baseline is inspired by Table 10, whose 40-column width was selected for larger workloads. Results establish an improvement relative to that explicit reference, not superiority over the best manually configured tiny circuit.

No held-out observations or real candidate proving times enter selection. The earlier `results/tuning.json` records preliminary 40-column empirical experiments only and is not consumed by this optimizer or the final benchmark.

Calibration is measured on the same machine with four Rayon threads. Each operation/domain combination uses 0.5 seconds of warmup, a two-second target measurement window and 30 Criterion samples. The coefficient is Criterion's mean point estimate, matching the original summary script. Raw samples and estimates are retained in `results/calibration/raw/`. Compilation is excluded from calibration time.

`results/optimizer.json` separately records total optimization wall time, feasibility-check time, estimator-call time and calibration time. The additional checks make optimization wall time unsuitable for a direct comparison with the paper's optimizer latency.

On the recorded machine, calibration took 87.9 seconds. The full search took 1122.1 seconds, of which 1110.5 seconds were the conservative feasibility checks and 6.23 seconds were calls to the original estimator. The remaining time includes logical generation and orchestration. The baseline was selected at 40 columns, implementation 2, k=11; the optimized circuit uses 10 columns, implementation 2, k=11. Their estimated proving costs are 3.082 seconds and 0.836 seconds, respectively. These are estimates, not final measured proof latencies.

The two selected models are frozen before `scripts/benchmark.py` runs. Both process the same ten seeded held-out observations in two blocks, with order fixed/optimized followed by optimized/fixed. Each block has one discarded warmup and reuses its proving key. Setup, key generation, input/public-value preparation, proof generation and verification are timed separately. Process RSS includes setup; it is not isolated prover memory.

The measured median proving times were 2.2862 s (fixed) and 0.7058 s (optimized): a 3.239× speedup and 69.13% latency reduction. Median verification times were 17.490 ms and 6.765 ms; proofs were 18,144 and 5,024 bytes. All ten paired outputs matched the same fixed-point reference and both configurations rejected modified scores, weights and proof bytes. The original estimator overpredicted both absolute latencies (3.082 s and 0.836 s); estimates must not be substituted for measured results.

The one-time calibration and conservative search dominate the cost of a single inference. These results support repeated use of the selected circuit, not a claim that running the entire optimization workflow for one score is cheaper.

## Commands

```sh
make build
make calibrate
make optimize
make benchmark
make test
```

Calibration is hardware-specific. On another machine, regenerate calibration, selection and final benchmark together. Do not reuse the included coefficients as universal performance constants.
