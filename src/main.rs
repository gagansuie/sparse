use std::fs;
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tenpak::{
    compress_bundle_with_codec, create_delta_artifact, decompress_bundle, materialize_artifact,
    Artifact, Bundle, CODEC_INT4_SYM_V1, CODEC_INT8_SYM_V1,
};

/// 10pak CLI: compress and decompress simple tensor bundles.
#[derive(Parser, Debug)]
#[command(author, version, about = "10pak model artifact compressor", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compress a JSON float tensor bundle into a binary 10pak artifact.
    Compress {
        /// Input JSON file describing a FloatBundle
        #[arg(short, long)]
        input: String,
        /// Output binary file for the compressed artifact
        #[arg(short, long)]
        output: String,
        /// Codec to use (e.g., int8_sym_v1, int4_sym_v1)
        #[arg(long, default_value = CODEC_INT8_SYM_V1)]
        codec: String,
    },
    /// Decompress a binary 10pak artifact back into a JSON bundle.
    Decompress {
        /// Input binary artifact file
        #[arg(short, long)]
        input: String,
        /// Output JSON file for the reconstructed FloatBundle
        #[arg(short, long)]
        output: String,
    },
    /// Inspect an artifact and print a human-readable summary.
    Inspect {
        /// Input binary artifact file
        #[arg(short, long)]
        input: String,
    },
    /// Plan compression by comparing codecs under an error constraint.
    Plan {
        /// Input JSON file describing a FloatBundle
        #[arg(short, long)]
        input: String,
        /// Output JSON plan file
        #[arg(short, long)]
        output: String,
        /// Maximum allowed mean absolute error (MAE)
        #[arg(long, default_value_t = 0.01)]
        max_mae: f64,
    },
    /// Benchmark codecs and print a size/error table.
    Bench {
        /// Input JSON file describing a FloatBundle
        #[arg(short, long)]
        input: String,
    },
    /// Create a delta artifact between a base artifact and a variant bundle.
    Delta {
        /// Base binary artifact file
        #[arg(long)]
        base: String,
        /// Variant JSON bundle file
        #[arg(long)]
        variant: String,
        /// Output binary delta artifact file
        #[arg(short, long)]
        output: String,
        /// L1 difference threshold to consider a tensor changed
        #[arg(long, default_value_t = 1e-3)]
        epsilon: f32,
    },
    /// Materialize a full artifact from a base artifact and a delta artifact.
    Materialize {
        /// Base binary artifact file
        #[arg(long)]
        base: String,
        /// Delta binary artifact file
        #[arg(long)]
        delta: String,
        /// Output binary artifact file
        #[arg(short, long)]
        output: String,
    },
    /// Run the full evaluation pipeline (downloads models, computes metrics, prints updated README)
    #[command(name = "runeval")]
    RunEval {
        /// Model to evaluate (default: gpt2)
        #[arg(long, default_value = "gpt2")]
        model: String,
        /// Number of evaluation samples (default: 128)
        #[arg(long, default_value_t = 128)]
        samples: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress {
            input,
            output,
            codec,
        } => compress_cmd(&input, &output, &codec)?,
        Commands::Decompress { input, output } => decompress_cmd(&input, &output)?,
        Commands::Inspect { input } => inspect_cmd(&input)?,
        Commands::Plan {
            input,
            output,
            max_mae,
        } => plan_cmd(&input, &output, max_mae)?,
        Commands::Bench { input } => bench_cmd(&input)?,
        Commands::Delta {
            base,
            variant,
            output,
            epsilon,
        } => delta_cmd(&base, &variant, &output, epsilon)?,
        Commands::Materialize {
            base,
            delta,
            output,
        } => materialize_cmd(&base, &delta, &output)?,
        Commands::RunEval { model, samples } => runeval_cmd(&model, samples)?,
    }

    Ok(())
}

#[derive(serde::Serialize)]
struct Candidate {
    codec: String,
    artifact_bytes: usize,
    mse: f64,
    mae: f64,
    max: f64,
}

fn evaluate_codecs(bundle: &Bundle) -> Result<Vec<Candidate>> {
    let codecs = [CODEC_INT8_SYM_V1, CODEC_INT4_SYM_V1];
    let mut candidates: Vec<Candidate> = Vec::new();

    for &codec in &codecs {
        let artifact = match compress_bundle_with_codec(bundle, codec) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Skipping codec {} due to compression error: {}", codec, e);
                continue;
            }
        };

        let bytes = bincode::serialize(&artifact).context("Failed to serialize artifact")?;
        let restored = match decompress_bundle(&artifact) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping codec {} due to decompression error: {}", codec, e);
                continue;
            }
        };

        let mut total = 0usize;
        let mut mse_acc = 0.0f64;
        let mut mae_acc = 0.0f64;
        let mut max_err = 0.0f64;

        for (orig_t, rec_t) in bundle.tensors.iter().zip(restored.tensors.iter()) {
            if orig_t.name != rec_t.name || orig_t.shape != rec_t.shape {
                continue;
            }
            for (o, r) in orig_t.data.iter().zip(rec_t.data.iter()) {
                let diff = (*o as f64 - *r as f64).abs();
                mse_acc += diff * diff;
                mae_acc += diff;
                if diff > max_err {
                    max_err = diff;
                }
                total += 1;
            }
        }

        if total == 0 {
            continue;
        }

        let mse = mse_acc / total as f64;
        let mae = mae_acc / total as f64;

        candidates.push(Candidate {
            codec: codec.to_string(),
            artifact_bytes: bytes.len(),
            mse,
            mae,
            max: max_err,
        });
    }

    Ok(candidates)
}

fn plan_cmd(input: &str, output: &str, max_mae: f64) -> Result<()> {
    let data = fs::read_to_string(input)
        .with_context(|| format!("Failed to read input JSON bundle: {}", input))?;

    let bundle: Bundle = serde_json::from_str(&data).context("Failed to parse JSON bundle")?;

    let candidates = evaluate_codecs(&bundle)?;

    if candidates.is_empty() {
        return Err(anyhow::Error::msg("No viable codecs evaluated in plan"));
    }

    // Choose smallest artifact among candidates that satisfy max_mae; if none,
    // choose the one with the lowest MAE.
    let mut best: Option<&Candidate> = None;
    for c in &candidates {
        if c.mae <= max_mae {
            best = match best {
                None => Some(c),
                Some(b) if c.artifact_bytes < b.artifact_bytes => Some(c),
                other => other,
            };
        }
    }

    if best.is_none() {
        for c in &candidates {
            best = match best {
                None => Some(c),
                Some(b) if c.mae < b.mae => Some(c),
                other => other,
            };
        }
    }

    let best = best.expect("candidates not empty");
    let best_codec = best.codec.clone();

    #[derive(serde::Serialize)]
    struct Plan {
        max_mae: f64,
        chosen_codec: String,
        candidates: Vec<Candidate>,
    }

    let plan = Plan {
        max_mae,
        chosen_codec: best_codec.clone(),
        candidates,
    };

    let json = serde_json::to_string_pretty(&plan).context("Failed to serialize plan JSON")?;
    fs::write(output, json).with_context(|| format!("Failed to write plan file: {}", output))?;

    println!("Plan written to {} (chosen codec: {})", output, best_codec);

    Ok(())
}

fn bench_cmd(input: &str) -> Result<()> {
    let data = fs::read_to_string(input)
        .with_context(|| format!("Failed to read input JSON bundle: {}", input))?;

    let bundle: Bundle = serde_json::from_str(&data).context("Failed to parse JSON bundle")?;

    let candidates = evaluate_codecs(&bundle)?;

    if candidates.is_empty() {
        return Err(anyhow::Error::msg("No viable codecs evaluated in bench"));
    }

    println!("Codec benchmark for {}:", input);
    println!(
        "  {:<14} {:>12} {:>12} {:>12} {:>12}",
        "codec", "bytes", "MSE", "MAE", "MAX"
    );
    for c in &candidates {
        println!(
            "  {:<14} {:>12} {:>12.6} {:>12.6} {:>12.6}",
            c.codec, c.artifact_bytes, c.mse, c.mae, c.max
        );
    }

    Ok(())
}

fn inspect_cmd(input: &str) -> Result<()> {
    let bytes =
        fs::read(input).with_context(|| format!("Failed to read artifact file: {}", input))?;

    let artifact: Artifact =
        bincode::deserialize(&bytes).context("Failed to deserialize artifact")?;

    println!("Artifact: {}", input);
    println!("  Version: {}", artifact.version);
    println!("  Codec:   {}", artifact.codec);
    println!("  Tensors: {}", artifact.tensors.len());

    for t in &artifact.tensors {
        let len = t.data.len();
        println!("    - {}: shape {:?}, {} values", t.name, t.shape, len);
    }

    Ok(())
}

fn compress_cmd(input: &str, output: &str, codec: &str) -> Result<()> {
    let data = fs::read_to_string(input)
        .with_context(|| format!("Failed to read input JSON bundle: {}", input))?;

    let bundle: Bundle = serde_json::from_str(&data).context("Failed to parse JSON bundle")?;
    let artifact: Artifact = compress_bundle_with_codec(&bundle, codec)
        .map_err(|e| anyhow::Error::msg(format!("Compression failed: {}", e)))?;

    let bytes = bincode::serialize(&artifact).context("Failed to serialize artifact")?;
    fs::write(output, &bytes)
        .with_context(|| format!("Failed to write artifact file: {}", output))?;

    let input_bytes = data.len();
    let output_bytes = bytes.len();
    println!("Input JSON size:   {} bytes", input_bytes);
    println!("Artifact size:     {} bytes", output_bytes);
    if output_bytes > 0 {
        let ratio = input_bytes as f64 / output_bytes as f64;
        println!("Compression ratio: {:.2}x (input / artifact)", ratio);
    }

    // Compute reconstruction error metrics by decompressing and comparing.
    let restored = decompress_bundle(&artifact)
        .map_err(|e| anyhow::Error::msg(format!("Decompression for metrics failed: {}", e)))?;

    let mut total = 0usize;
    let mut mse_acc = 0.0f64;
    let mut mae_acc = 0.0f64;
    let mut max_err = 0.0f64;

    for (orig_t, rec_t) in bundle.tensors.iter().zip(restored.tensors.iter()) {
        if orig_t.name != rec_t.name || orig_t.shape != rec_t.shape {
            continue;
        }
        for (o, r) in orig_t.data.iter().zip(rec_t.data.iter()) {
            let diff = (*o as f64 - *r as f64).abs();
            mse_acc += diff * diff;
            mae_acc += diff;
            if diff > max_err {
                max_err = diff;
            }
            total += 1;
        }
    }

    if total > 0 {
        let mse = mse_acc / total as f64;
        let mae = mae_acc / total as f64;
        println!("Reconstruction error (over {} values):", total);
        println!("  MSE:  {:.6}", mse);
        println!("  MAE:  {:.6}", mae);
        println!("  MAX:  {:.6}", max_err);
    }

    Ok(())
}

fn decompress_cmd(input: &str, output: &str) -> Result<()> {
    let bytes =
        fs::read(input).with_context(|| format!("Failed to read artifact file: {}", input))?;

    let artifact: Artifact =
        bincode::deserialize(&bytes).context("Failed to deserialize artifact")?;
    let bundle = decompress_bundle(&artifact)
        .map_err(|e| anyhow::Error::msg(format!("Decompression failed: {}", e)))?;

    let json = serde_json::to_string_pretty(&bundle).context("Failed to serialize JSON bundle")?;
    fs::write(output, json)
        .with_context(|| format!("Failed to write JSON bundle file: {}", output))?;

    Ok(())
}

fn delta_cmd(base: &str, variant: &str, output: &str, epsilon: f32) -> Result<()> {
    let base_bytes =
        fs::read(base).with_context(|| format!("Failed to read base artifact file: {}", base))?;
    let base_artifact: Artifact =
        bincode::deserialize(&base_bytes).context("Failed to deserialize base artifact")?;

    let variant_json = fs::read_to_string(variant)
        .with_context(|| format!("Failed to read variant JSON bundle: {}", variant))?;
    let variant_bundle: Bundle =
        serde_json::from_str(&variant_json).context("Failed to parse variant JSON bundle")?;

    let delta_artifact = create_delta_artifact(&base_artifact, &variant_bundle, epsilon)
        .map_err(|e| anyhow::Error::msg(format!("Delta creation failed: {}", e)))?;

    let delta_bytes =
        bincode::serialize(&delta_artifact).context("Failed to serialize delta artifact")?;
    fs::write(output, &delta_bytes)
        .with_context(|| format!("Failed to write delta artifact file: {}", output))?;

    println!(
        "Delta artifact written to {} (base: {}, variant: {}, epsilon: {})",
        output, base, variant, epsilon
    );

    Ok(())
}

fn runeval_cmd(model: &str, samples: usize) -> Result<()> {
    println!("[tenpak] Starting evaluation pipeline");
    println!("[tenpak] Model: {}", model);
    println!("[tenpak] Samples: {}", samples);

    // Find the binary location
    let exe = std::env::current_exe().context("Failed to get executable path")?;
    let exe_dir = exe.parent().context("Failed to get executable directory")?;

    // Determine root directory based on where binary is located
    // For cargo build: binary is in target/release/tenpak, root is ../..
    // For packaged: binary is in bin/tenpak, root is ..
    let root_dir = if exe_dir.ends_with("target/release") || exe_dir.ends_with("target\\release") {
        // Running from cargo build - go up two levels
        exe_dir
            .parent()
            .and_then(|p| p.parent())
            .context("Failed to find project root")?
            .to_path_buf()
    } else if exe_dir.ends_with("bin") {
        // Running from packaged binary - go up one level
        exe_dir
            .parent()
            .context("Failed to find project root")?
            .to_path_buf()
    } else {
        // Fallback: assume current directory
        std::env::current_dir().context("Failed to get current directory")?
    };

    println!("[tenpak] Project root: {}", root_dir.display());

    // Check for Python
    let python = find_python()?;
    println!("[tenpak] Using Python: {}", python);

    // Look for scripts directory
    let scripts_dir = root_dir.join("scripts");
    if !scripts_dir.exists() {
        return Err(anyhow::Error::msg(
            format!("Scripts directory not found at: {}\nMake sure the tenpak package includes the scripts/ directory.", scripts_dir.display())
        ));
    }

    // Check if download script exists
    let download_script = scripts_dir.join("download_demo_models.sh");
    if !download_script.exists() {
        return Err(anyhow::Error::msg(format!(
            "Download script not found at: {}",
            download_script.display()
        )));
    }

    // Run download script
    println!("[tenpak] Downloading models...");
    let status = Command::new("bash")
        .arg(&download_script)
        .current_dir(&root_dir)
        .env("MODELS", model)
        .status()
        .context("Failed to run download script")?;

    if !status.success() {
        return Err(anyhow::Error::msg("Model download failed"));
    }

    // Check if eval script exists
    let eval_script = scripts_dir.join("run_eval_and_update_readme.py");
    if !eval_script.exists() {
        return Err(anyhow::Error::msg(format!(
            "Evaluation script not found at: {}",
            eval_script.display()
        )));
    }

    // Run evaluation script
    println!("[tenpak] Running evaluation...");
    let status = Command::new(&python)
        .arg(&eval_script)
        .current_dir(&root_dir)
        .env("TENPAK_EVAL_MODEL", model)
        .env("TENPAK_EVAL_SAMPLES", samples.to_string())
        .env("TENPAK_BIN", exe.to_string_lossy().to_string())
        .status()
        .context("Failed to run evaluation script")?;

    if !status.success() {
        return Err(anyhow::Error::msg("Evaluation failed"));
    }

    println!("[tenpak] Evaluation complete!");

    Ok(())
}

fn find_python() -> Result<String> {
    let candidates = ["python3", "python"];

    for cmd in &candidates {
        if Command::new(cmd)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Ok(cmd.to_string());
        }
    }

    Err(anyhow::Error::msg(
        "Python not found. Please install Python 3 (python3 or python must be in PATH)",
    ))
}

fn materialize_cmd(base: &str, delta: &str, output: &str) -> Result<()> {
    let base_bytes =
        fs::read(base).with_context(|| format!("Failed to read base artifact file: {}", base))?;
    let base_artifact: Artifact =
        bincode::deserialize(&base_bytes).context("Failed to deserialize base artifact")?;

    let delta_bytes = fs::read(delta)
        .with_context(|| format!("Failed to read delta artifact file: {}", delta))?;
    let delta_artifact: Artifact =
        bincode::deserialize(&delta_bytes).context("Failed to deserialize delta artifact")?;

    let merged_artifact = materialize_artifact(&base_artifact, &delta_artifact)
        .map_err(|e| anyhow::Error::msg(format!("Materialization failed: {}", e)))?;

    let merged_bytes = bincode::serialize(&merged_artifact)
        .context("Failed to serialize materialized artifact")?;
    fs::write(output, &merged_bytes)
        .with_context(|| format!("Failed to write materialized artifact file: {}", output))?;

    println!(
        "Materialized artifact written to {} (base: {}, delta: {})",
        output, base, delta
    );

    Ok(())
}
