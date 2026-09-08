//! Minimal application harness. All inference constraints come from upstream ZKML.
use halo2_proofs::{
    dev::MockProver,
    halo2curves::{
        bn256::{Bn256, Fr, G1Affine},
        ff::Field,
    },
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, VerifyingKey},
    poly::{
        commitment::Params,
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
    SerdeFormat,
};
use serde_json::json;
use std::{env, fs, io::BufReader, path::Path, time::Instant};
use zkml::{
    model::ModelCircuit,
    utils::{helpers::get_public_values, loader::load_config_msgpack},
};

fn signed(x: i64) -> Fr {
    if x < 0 {
        -Fr::from(x.unsigned_abs())
    } else {
        Fr::from(x as u64)
    }
}

fn model_values(model: &str) -> Vec<Fr> {
    let config = load_config_msgpack(model);
    assert!(config.commit_before.as_ref().map_or(true, |v| v.is_empty()));
    assert!(config.commit_after.as_ref().map_or(true, |v| v.is_empty()));
    let outputs = &config.out_idxes;
    assert!(
        outputs.len() >= 2,
        "Public model weights must precede the score"
    );
    outputs[..outputs.len() - 1]
        .iter()
        .flat_map(|id| {
            config
                .tensors
                .iter()
                .find(|t| t.idx == *id)
                .expect("Missing public model tensor")
                .data
                .iter()
                .map(|x| signed(*x))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn check_model(public: &[Fr], expected: &[Fr]) {
    assert_eq!(public.len(), expected.len() + 1);
    assert_eq!(
        &public[..expected.len()],
        expected,
        "Unexpected model weights"
    );
}

fn score(public: &[Fr]) -> u64 {
    let bytes = public.last().unwrap().to_bytes();
    assert!(bytes[8..].iter().all(|b| *b == 0));
    u64::from_le_bytes(bytes[..8].try_into().unwrap())
}

fn accepted(
    params: &ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    public: &[Fr],
    proof: &[u8],
) -> bool {
    let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof);
    verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<'_, Bn256>,
        Challenge255<G1Affine>,
        _,
        _,
    >(
        params,
        vk,
        SingleStrategy::new(params),
        &[&[public]],
        &mut transcript,
    )
    .is_ok()
}

fn peak_rss_bytes() -> i64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) }, 0);
    if cfg!(target_os = "macos") {
        usage.ru_maxrss as i64
    } else {
        usage.ru_maxrss as i64 * 1024
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() >= 5, "Usage: credit-zk mock|prove MODEL INPUTS_JSON OUT_DIR; credit-zk verify MODEL RUN_DIR INDEX");
    let mode = &args[1];
    let model = &args[2];
    let cfg = load_config_msgpack(model);
    let expected = model_values(model);
    if mode == "verify" {
        // Configuration and verification key are trusted application artifacts; no witness is loaded.
        let dir = Path::new(&args[3]);
        let index: usize = args[4].parse().unwrap();
        let _circuit = ModelCircuit::<Fr>::generate_from_msgpack(cfg.clone(), false);
        let params = ParamsKZG::<Bn256>::read(&mut BufReader::new(
            fs::File::open(dir.join("params.bin")).unwrap(),
        ))
        .unwrap();
        let vk = VerifyingKey::read::<_, ModelCircuit<Fr>>(
            &mut BufReader::new(fs::File::open(dir.join("vk.bin")).unwrap()),
            SerdeFormat::RawBytes,
            (),
        )
        .unwrap();
        let raw = fs::read(dir.join(format!("public_{index}.bin"))).unwrap();
        assert_eq!(raw.len() % 32, 0);
        let public: Vec<Fr> = raw
            .chunks(32)
            .map(|x| Option::<Fr>::from(Fr::from_bytes(x.try_into().unwrap())).unwrap())
            .collect();
        check_model(&public, &expected);
        let proof = fs::read(dir.join(format!("proof_{index}.bin"))).unwrap();
        assert!(accepted(&params, &vk, &public, &proof), "Proof rejected");
        println!(
            "{}",
            json!({"verified": true, "score_integer": score(&public), "scale_factor": cfg.global_sf})
        );
        return;
    }
    let inputs: Vec<String> = serde_json::from_slice(&fs::read(&args[3]).unwrap()).unwrap();
    assert!(!inputs.is_empty());
    let out = Path::new(&args[4]);
    fs::create_dir_all(out).unwrap();
    let first = ModelCircuit::<Fr>::generate_from_file(model, &inputs[0]);
    if mode == "mock" {
        let mut rows = vec![];
        for input in &inputs {
            let circuit = ModelCircuit::<Fr>::generate_from_file(model, input);
            drop(MockProver::run(cfg.k as u32, &circuit, vec![vec![]]).unwrap());
            let public = get_public_values::<Fr>();
            check_model(&public, &expected);
            let prover = MockProver::run(cfg.k as u32, &circuit, vec![public.clone()]).unwrap();
            assert_eq!(prover.verify(), Ok(()));
            let mut bad_score = public.clone();
            *bad_score.last_mut().unwrap() += Fr::ONE;
            assert!(MockProver::run(cfg.k as u32, &circuit, vec![bad_score])
                .unwrap()
                .verify()
                .is_err());
            let mut bad_weight = public.clone();
            bad_weight[0] += Fr::ONE;
            assert!(MockProver::run(cfg.k as u32, &circuit, vec![bad_weight])
                .unwrap()
                .verify()
                .is_err());
            rows.push(json!({"input": input, "score_integer": score(&public),
                "valid": true, "wrong_score_rejected": true, "wrong_weight_rejected": true}));
        }
        fs::write(
            out.join("mock.json"),
            serde_json::to_vec_pretty(&rows).unwrap(),
        )
        .unwrap();
        return;
    }
    assert_eq!(mode, "prove");
    // Locally generated experimental SRS, not a multiparty setup.
    let start = Instant::now();
    let params = ParamsKZG::<Bn256>::setup(cfg.k as u32, rand::thread_rng());
    let setup_s = start.elapsed().as_secs_f64();
    params
        .write(&mut fs::File::create(out.join("params.bin")).unwrap())
        .unwrap();
    let start = Instant::now();
    let vk = keygen_vk(&params, &first).unwrap();
    let vk_s = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let pk = keygen_pk(&params, vk, &first).unwrap();
    let pk_s = start.elapsed().as_secs_f64();
    let vk_bytes = pk.get_vk().to_bytes(SerdeFormat::RawBytes);
    let pk_bytes = pk.to_bytes(SerdeFormat::RawBytes);
    fs::write(out.join("vk.bin"), &vk_bytes).unwrap();
    // Record proving-key size without retaining another large disk artifact.
    let pk_size = pk_bytes.len();
    drop(pk_bytes);
    let mut rows = vec![];
    // One unmeasured warmup followed by the specified observations; keys are reused.
    for (iteration, input) in std::iter::once(&inputs[0]).chain(inputs.iter()).enumerate() {
        let circuit = ModelCircuit::<Fr>::generate_from_file(model, input);
        let start = Instant::now();
        drop(MockProver::run(cfg.k as u32, &circuit, vec![vec![]]).unwrap());
        let public = get_public_values::<Fr>();
        check_model(&public, &expected);
        let preparation_s = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            Challenge255<G1Affine>,
            _,
            _,
            ModelCircuit<Fr>,
        >(
            &params,
            &pk,
            &[circuit],
            &[&[&public]],
            rand::thread_rng(),
            &mut transcript,
        )
        .unwrap();
        let proof = transcript.finalize();
        let proving_s = start.elapsed().as_secs_f64();
        let start = Instant::now();
        assert!(accepted(&params, pk.get_vk(), &public, &proof));
        let verification_s = start.elapsed().as_secs_f64();
        let mut bad_score = public.clone();
        *bad_score.last_mut().unwrap() += Fr::ONE;
        assert!(!accepted(&params, pk.get_vk(), &bad_score, &proof));
        let mut bad_weight = public.clone();
        bad_weight[0] += Fr::ONE;
        assert!(!accepted(&params, pk.get_vk(), &bad_weight, &proof));
        let mut bad_proof = proof.clone();
        bad_proof[0] ^= 1;
        assert!(!accepted(&params, pk.get_vk(), &public, &bad_proof));
        if iteration == 0 {
            continue;
        }
        let index = iteration - 1;
        fs::write(out.join(format!("proof_{index}.bin")), &proof).unwrap();
        let raw: Vec<u8> = public.iter().flat_map(|v| v.to_bytes()).collect();
        fs::write(out.join(format!("public_{index}.bin")), raw).unwrap();
        rows.push(json!({"index": index, "input": input, "proving_s": proving_s,
            "verification_s": verification_s, "public_value_preparation_s": preparation_s,
            "proof_bytes": proof.len(), "score_integer": score(&public), "verified": true,
            "wrong_score_rejected": true, "wrong_weight_rejected": true, "altered_proof_rejected": true}));
    }
    let report = json!({"k": cfg.k, "columns": cfg.num_cols, "scale_factor": cfg.global_sf,
        "setup_s": setup_s, "verification_key_s": vk_s, "proving_key_s": pk_s,
        "verification_key_bytes": vk_bytes.len(), "proving_key_bytes": pk_size,
        "peak_process_rss_bytes": peak_rss_bytes(), "warmup_proofs": 1, "measurements": rows});
    fs::write(
        out.join("measurements.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{}", report);
}
