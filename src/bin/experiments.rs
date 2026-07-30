//! # Experiments entry point.
//! 
//! Entry point for the benchmark.
//! 
//! The `result_path` parameter must be a directory, as this is where the results will be saved, with a new .csv 
//! file being created. The results of each run will be saved in this same directory for validation in `checker.py`, 
//! but these results will be deleted after use.

use clap::{Parser, Subcommand};
use pso_for_order_batching_rust::{Problem, sbpso, methods::SolverOutput};
use std::path::PathBuf;

/// This program reads the instances and executes the benchmark for the specified method.
#[derive(Parser)]
struct Args {
    /// Path to the directory containing the instances.
    dataset_path: String,

    /// Path to the directory where the results will be saved.
    result_path: String,

    /// Path to the seeds file.
    seeds_path: String,

    #[command(subcommand)]
    method: Method,
}

/// Parameters for the implemented methods. The default values are the same as those used in the article.
#[derive(Subcommand)]
enum Method {
    /// Parameters for Set-Based Particle Swarm Optimization.
    Sbpso {
        /// Population size.
        #[arg(short, long, default_value_t = 50)]
        size: usize,

        /// Maximum number of iterations.
        #[arg(short, long, default_value_t = 600)]
        iterations: usize,

        /// Cognitive component.
        #[arg(long, default_value_t = 0.9297)]
        c1: f64,

        /// Social component.
        #[arg(long, default_value_t = 0.2266)]
        c2: f64,

        /// Random addition coefficient.
        #[arg(long, default_value_t = 1.3086)]
        c3: f64,

        /// Random removal coefficient.
        #[arg(long, default_value_t = 2.1526)]
        c4: f64,

        /// Tournament size for k-tournament selection.
        #[arg(short, default_value_t = 7)]
        k: usize,
    },
    /// Parameters for Modified Binary Particle Swarm Optimization.
    Mbpso {
        /// Population size.
        #[arg(short, long, default_value_t = 50)]
        size: usize,

        /// Maximum number of iterations.
        #[arg(short, long, default_value_t = 600)]
        iterations: usize,

        /// Initial inertia.
        #[arg(long, default_value_t = 0.9)]
        w_max: f64,

        /// Final inertia.
        #[arg(long, default_value_t = 0.4)]
        w_min: f64,

        /// Cognitive component.
        #[arg(long, default_value_t = 2.0)]
        c1: f64,

        /// Social component.
        #[arg(long, default_value_t = 2.0)]
        c2: f64,

        /// Maximum velocity.
        #[arg(long, default_value_t = 6.0)]
        v_max: f64,

        /// Minumum velocity.
        #[arg(long, default_value_t = -6.0)]
        v_min: f64,

        /// Probability of mutation.
        #[arg(short, long, default_value_t = 0.05)]
        r_mu: f64,
    },
    /// Parameters for Modified Binary Particle Swarm Optimization with a parameterized transfer function.
    Mbpsozt {
        /// Population size.
        #[arg(short, long, default_value_t = 50)]
        size: usize,

        /// Maximum number of iterations.
        #[arg(short, long, default_value_t = 600)]
        iterations: usize,

        /// Initial inertia.
        #[arg(long, default_value_t = 0.9)]
        w_max: f64,

        /// Final inertia.
        #[arg(long, default_value_t = 0.4)]
        w_min: f64,

        /// Cognitive component.
        #[arg(long, default_value_t = 2.0)]
        c1: f64,

        /// Social component.
        #[arg(long, default_value_t = 2.0)]
        c2: f64,

        /// Maximum velocity.
        #[arg(long, default_value_t = 6.0)]
        v_max: f64,

        /// Minumum velocity.
        #[arg(long, default_value_t = -6.0)]
        v_min: f64,

        /// Probability of mutation.
        #[arg(short, long, default_value_t = 0.05)]
        r_mu: f64,

        /// Transfer function parameter.
        #[arg(short, default_value_t = 0.5)]
        k: f64,
    },
}

/// Results by instance.
#[derive(Clone, Copy)]
struct ExperimentResult {
    min: f64,
    max: f64,
    mean: f64,
    std_dev: f64,
    mean_time: f64,
}

impl ExperimentResult {
    /// Creates a single `ExperimentResult` instance based on the values returned by the method for all seeds.
    fn from_outputs(outputs: Vec<SolverOutput>) -> Self {
        let objectives: Vec<f64> = outputs
            .iter()
            .map(|o| o.convergence.last().copied().unwrap_or(f64::NEG_INFINITY))
            .collect();
        let times: Vec<f64> = outputs.iter().map(|o| o.time).collect();

        let mean: f64 = objectives.iter().sum::<f64>() / objectives.len() as f64;
        let variance: f64 = objectives.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / objectives.len() as f64;

        ExperimentResult {
            min: objectives.iter().copied().fold(f64::INFINITY, f64::min),
            max: objectives.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            mean,
            std_dev: variance.sqrt(),
            mean_time: times.iter().sum::<f64>() / times.len() as f64,
        }
    }
}

/// Reads the file containing the seeds and returns a vector with the values read.
fn read_seeds(path: &str) -> anyhow::Result<Vec<u64>> {
    let content: String = std::fs::read_to_string(path)?;
    
    let seeds: Vec<u64> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            line.trim()
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("line {}: '{}' is not a valid u64", i + 1, line.trim()))
        })
        .collect::<anyhow::Result<Vec<u64>>>()?;

    Ok(seeds)
}

/// Builds a vector containing the paths of all files in the specified directory.
fn read_instances(dir: &str) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();

    paths.sort_unstable();

    Ok(paths)
}

/// Creates and populates the results CSV file based on the received values. 
/// 
/// `results` is a vector containing the results obtained (`ExperimentResult`) for each instance (`String`).
fn write_results(results: &[(String, ExperimentResult)], output: PathBuf) -> anyhow::Result<()> {
    let mut wtr: csv::Writer<std::fs::File> = csv::Writer::from_path(output)?;
    wtr.write_record(&["dataset", "min", "max", "mean", "std_dev", "mean_time"])?;

    for (dataset, result) in results {
        wtr.write_record(&[
            dataset,
            &result.min.to_string(),
            &result.max.to_string(),
            &result.mean.to_string(),
            &result.std_dev.to_string(),
            &result.mean_time.to_string(),
        ])?;
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();
    
    let seeds: Vec<u64> = read_seeds(&args.seeds_path)?;
    let instances: Vec<PathBuf> = read_instances(&args.dataset_path)?;

    let method_name: &str = match args.method {
        Method::Sbpso { .. } => "sbpso",
        Method::Mbpso { .. } => "mbpso",
        Method::Mbpsozt { .. } => "mbpsozt",
    };

    let mut results: Vec<(String, ExperimentResult)> = Vec::with_capacity(instances.len());
    for instance in &instances {
        let instance_str: &str = instance.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {:?}", instance))?;
        
        let file_name: String = instance.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("invalid file name: {:?}", instance))?
            .to_string();

        let problem: Problem = Problem::new(instance_str)?;
        let mut experiments: Vec<SolverOutput> = Vec::with_capacity(seeds.len());
        for &seed in &seeds {
            let output: SolverOutput = match args.method {
                Method::Sbpso { size, iterations, c1, c2, c3, c4, k } => {
                    sbpso(&problem, seed, size, iterations, c1, c2, c3, c4, k)?
                },
                Method::Mbpso { .. } => { 
                    todo!("MBPSO not yet implemented.")
                },
                Method::Mbpsozt { .. } => {
                    todo!("MBPSOzt not yet implemented.")
                }
            };

            experiments.push(output);
        }

        results.push((file_name, ExperimentResult::from_outputs(experiments)));
    }

    let mut result_path = PathBuf::from(&args.result_path);
    result_path.push(format!("{method_name}.csv"));
    write_results(&results, result_path)?;

    Ok(())
}