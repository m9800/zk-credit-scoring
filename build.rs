//! Compile upstream estimator/microbenchmarks with explicit, reviewable adapters.
//! The source submodule and the estimator's cost formulas remain unchanged.
use std::{env, fs, path::PathBuf};

fn replace_once(text: &str, old: &str, new: &str) -> String {
    assert_eq!(text.matches(old).count(), 1, "Upstream source changed: {old}");
    text.replacen(old, new, 1)
}

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let source = "vendor/zkml/src/bin/estimate_cost.rs";
    println!("cargo:rerun-if-changed={source}");
    let estimator = fs::read_to_string(source).unwrap();
    let estimator = replace_once(&estimator, "#![feature(int_roundings)]\n", "");
    // The original row estimate ignores numeric ranges and may undersize tiny
    // circuits. The caller supplies an independently MockProver-validated k.
    let estimator = replace_once(&estimator,
        "let k = (num_rows as f32).log2().ceil() as u64;",
        "let estimated_k = (num_rows as f32).log2().ceil() as u64;\n  println!(\"Arithmetic-row k estimate: {}\", estimated_k);\n  let k = estimated_k.max(circuit.k as u64);");
    // Missing calibration must never silently contribute zero cost.
    let estimator = replace_once(&estimator,
        "println!(\"Warning: k is out of range for {} time estimation. \", oper_type);\n    0.",
        "panic!(\"Missing calibration for {}_{} at k={}\", kzg_or_ipa, oper_type, k);");
    fs::write(out.join("estimate_cost.rs"), estimator).unwrap();
    for (operation, suffix) in [("fft", "fft"), ("msm", "msm"), ("add", "add"),
        ("mul", "mul"), ("permute", "permute_expression_pair")] {
        let source = format!("vendor/zkml/benches/benchmark_kzg_{suffix}.rs");
        println!("cargo:rerun-if-changed={source}");
        let bench = fs::read_to_string(source).unwrap();
        let bench = replace_once(&bench, "for k in 13..20 {", "for k in 10..15 {");
        fs::write(out.join(format!("{operation}.rs")), bench).unwrap();
    }
}
