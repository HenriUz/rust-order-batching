pub mod sbpso;

/// Results of the methods.
pub struct SolverOutput {
    pub convergence: Vec<f64>,
    pub time: f64
}

/// Method errors.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}