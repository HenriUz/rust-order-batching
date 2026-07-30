//! # Solver entry point.
//! 
//! Entry point for executing methods. It receives the input parameters and executes the corresponding method.

use clap::{Parser, Subcommand};
use pso_for_order_batching_rust::{Problem, sbpso};

/// This program reads the instance and executes the specified method to resolve it. Finally, the result is saved to the specified file.
#[derive(Parser)]
struct Args {
    /// Path to the instance file.
    dataset_path: String,

    /// Path to the result file.
    result_path: String,

    /// Seed number.
    seed: u64,

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

fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();
    let problem: Problem = Problem::new(&args.dataset_path)?;

    match args.method {
        Method::Sbpso { size, iterations, c1, c2, c3, c4, k } => {
            let result = sbpso(&problem, args.seed, size, iterations, c1, c2, c3, c4, k)?;
            println!("Got {} in {} seconds.", result.convergence.last().unwrap_or(&f64::NEG_INFINITY), result.time);
        },
        Method::Mbpso { .. } => { 
            todo!("MBPSO not yet implemented.")
        },
        Method::Mbpsozt { .. } => {
            todo!("MBPSOzt not yet implemented.")
        }
    }

    Ok(())
}