use rand::{RngExt, rngs::StdRng, SeedableRng, seq::IteratorRandom};
use std::{collections::HashSet};
use std::time::Instant;

use super::{SolverError, SolverOutput};
use crate::Problem;

/// Represents a particle in the swarm.
struct Particle {
    aisles_items: Vec<u32>,
    x: HashSet<usize>,
    objective: f64,
    pbest: HashSet<usize>,
    pbest_obj: f64
}

impl Particle {
    /// Creates a `Particle` with a random initial position.
    /// 
    /// Each aisle has a 50% chance of being included in the initial solution.
    /// 
    /// # Note
    /// `pbest` and `pbest_obj` are left uninitialized (`HashSet::new()` and `f64::NEG_INFINITY`
    /// respectively). The caller is responsible for updating them,
    /// typically during the first pbest/gbest update step.
    fn new(problem: &Problem, rng: &mut StdRng) -> Self {
        let mut aisles_items: Vec<u32> = vec![0; problem.i];
        let mut x: HashSet<usize> = HashSet::new();

        for aisle in 0..problem.a {
            if rng.random_bool(0.5) {
                x.insert(aisle);
                for (&item, &qty) in &problem.aisles[aisle] {
                    aisles_items[item] += qty;
                }
            }
        };

        let objective: f64 = problem.objective_function(
            problem.add_orders(&aisles_items), x.len() as u32
        );

        Particle { aisles_items, x, objective, pbest: HashSet::new(), pbest_obj: f64::NEG_INFINITY }
    }

    /// Adds an aisle to the solution and updates the available items.
    fn insert_aisle(&mut self, aisle: usize, problem: &Problem) {
        if self.x.insert(aisle) {
            for (&item, &qty) in &problem.aisles[aisle] {
                self.aisles_items[item] += qty;
            }
        }
    }

    /// Removes an aisle from the solution and updates the available items.
    fn remove_aisle(&mut self, aisle: usize, problem: &Problem) {
        if self.x.remove(&aisle) {
            for (&item, &qty) in &problem.aisles[aisle] {
                self.aisles_items[item] -= qty;
            }
        }
    }

    /// Updates pbest to the current position and objective value.
    fn accept_current_as_pbest(&mut self) {
        self.pbest = self.x.clone();
        self.pbest_obj = self.objective;
    }
}

/// Represents the operation performed at the particle's velocity.
#[derive(PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
enum Op {
    Add,
    Remove
}

/// Multiplies a velocity by a scalar value.
///
/// Selects `floor(scalar * |velocity|)` random elements from `velocity`
/// and inserts them into `result_velocity`.
fn scalar_multiplication(
    rng: &mut StdRng,
    scalar: f64,
    velocity: &HashSet<(Op, usize)>,
    result_velocity: &mut HashSet<(Op, usize)>
) {
    // `scalar` is guaranteed to be in [0.0, 1.0] by the caller.
    let k: usize = (scalar * velocity.len() as f64) as usize;

    let mut sorted_velocity: Vec<&(Op, usize)> = velocity.iter().collect();
    sorted_velocity.sort_unstable();

    for &element in sorted_velocity.into_iter().sample(rng, k) {
        result_velocity.insert(element);
    }
}

/// Computes the velocity needed to transform `current` into `target`.
///
/// Elements only in `target` become additions (`Op::Add`);
/// elements only in `current` become removals (`Op::Remove`).
fn difference_in_positions(
    target: &HashSet<usize>,
    current: &HashSet<usize>
) -> HashSet<(Op, usize)> {
    let mut velocity: HashSet<(Op, usize)> = HashSet::with_capacity(target.len() + current.len());
    
    velocity.extend((target - current).into_iter().map(|aisle| (Op::Add, aisle)));
    velocity.extend((current - target).into_iter().map(|aisle| (Op::Remove, aisle)));

    velocity
}

/// Returns the number of elements to operate on in a velocity update step.
///
/// Uses stochastic rounding on `beta`: the result is `floor(beta)` with a
/// probability of `fract(beta)` of rounding up to `ceil(beta)`.
/// 
/// The result is clamped to `set.len()`.
fn number_of_elements(
    rng: &mut StdRng,
    beta: f64,
    set: &HashSet<usize>
) -> usize {
    let count: usize = beta as usize + if rng.random::<f64>() < beta.fract() { 1 } else { 0 };
    count.min(set.len())
}

/// Greedily selects `n_to_add` aisles from `candidates` using tournament selection.
///
/// For each addition, `k` candidates are sampled and the one that best improves
/// the objective is chosen. The selected elements are added to `current`.
fn k_tournament_selection(
    rng: &mut StdRng,
    problem: &Problem,
    particle: &Particle,
    current: &mut HashSet<(Op, usize)>,
    candidates: &HashSet<usize>,
    n_to_add: usize,
    k: usize
) {
    // Sorting to ensure reproducibility.
    let mut remaining: Vec<&usize> = candidates.iter().collect();
    remaining.sort_unstable();

    let mut current_items: Vec<u32> = particle.aisles_items.clone();
    let mut n_aisles: u32 = particle.x.len() as u32 + 1;

    for _ in 0..n_to_add {
        let tournament_size: usize = k.min(remaining.len());
        let contestants: Vec<&&usize> = remaining.iter().sample(rng, tournament_size);

        let mut best_aisle: Option<usize> = None;
        let mut best_obj: f64 = f64::NEG_INFINITY;

        for &&aisle in contestants {
            for (&item, &qty) in &problem.aisles[aisle] {
                current_items[item] += qty;
            }

            let objective: f64 = problem.objective_function(
                problem.add_orders(&current_items), n_aisles
            );

            if objective > best_obj {
                best_aisle = Some(aisle);
                best_obj = objective;
            }

            for (&item, &qty) in &problem.aisles[aisle] {
                current_items[item] -= qty;
            }
        }

        if let Some(best_aisle) = best_aisle {
            current.insert((Op::Add, best_aisle));
            
            // O(n) removal, but depending on the size of the vector, it’s more efficient than maintaining a HashMap for the indices.
            remaining.retain(|&&x| x != best_aisle);
    
            // Commit the changes for the next element.
            for (&item, &qty) in &problem.aisles[best_aisle] {
                current_items[item] += qty;
            }
            n_aisles += 1;
        }
    }
}

/// Randomly selects up to `n_to_remove` aisles from `consensus` to remove.
///
/// The selected elements are added to `current` as removals (`Op::Remove`).
fn removal_of_elements(
    rng: &mut StdRng,
    consensus: &HashSet<usize>,
    n_to_remove: usize,
    current: &mut HashSet<(Op, usize)>
) {
    let mut sorted: Vec<&usize> = consensus.iter().collect();
    sorted.sort_unstable();

    // `n_to_remove` is guaranteed to be less than or equal to the size of the consensus by the caller.
    for &aisle in sorted.into_iter().sample(rng, n_to_remove) {
        current.insert((Op::Remove, aisle));
    }
}

pub fn sbpso(
    problem: &Problem,
    seed: u64,
    size: usize,
    iterations: usize,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    k: usize,
) -> Result<SolverOutput, SolverError> {
    let check = |v: f64, name: &str| {
        if !(0.0..=1.0).contains(&v) {
            Err(SolverError::InvalidParameters(
                format!("{name} must be in [0, 1], got {v}")
            ))
        } else {
            Ok(())
        }
    };
    check(c1, "c1")?;
    check(c2, "c2")?;

    let start: Instant = Instant::now();
    let mut rng: StdRng = StdRng::seed_from_u64(seed);

    // Initializes the swarm and sets the global values.
    let universe: HashSet<usize> = (0..problem.a).collect();
    let mut gbest: HashSet<usize> = HashSet::new();
    let mut gbest_obj: f64 = f64::NEG_INFINITY;

    let mut swarm: Vec<Particle> = (0..size)
        .map(|_| Particle::new(problem, &mut rng))
        .collect();

    let mut convergence: Vec<f64> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        // Both loops are necessary to ensure that the position update always uses the best position found.
        for i in 0..size {
            if swarm[i].objective >= swarm[i].pbest_obj {
                swarm[i].accept_current_as_pbest();
            }
            if swarm[i].objective >= gbest_obj {
                gbest = swarm[i].x.clone();
                gbest_obj = swarm[i].objective;
            }
        }

        convergence.push(gbest_obj);

        for i in 0..size {
            let mut velocity: HashSet<(Op, usize)> = HashSet::new();

            let scalar: f64 = c1 * rng.random::<f64>();
            scalar_multiplication(
                &mut rng, scalar,
                &difference_in_positions(&swarm[i].pbest, &swarm[i].x), 
                &mut velocity
            );

            let scalar: f64 = c2 * rng.random::<f64>();
            scalar_multiplication(
                &mut rng, scalar, 
                &difference_in_positions(&gbest, &swarm[i].x),
                &mut velocity
            );

            let external_aisles: HashSet<usize> = universe.iter()
                .filter(|&&e| !swarm[i].x.contains(&e) && !swarm[i].pbest.contains(&e) && !gbest.contains(&e))
                .copied()
                .collect();

            let beta: f64 = c3 * rng.random::<f64>();
            let n_elements: usize = number_of_elements(&mut rng, beta, &external_aisles);
            k_tournament_selection(&mut rng, problem, &swarm[i], &mut velocity, &external_aisles, n_elements, k);

            let consensus_aisles: HashSet<usize> = swarm[i].x.iter()
                .filter(|&&e| swarm[i].pbest.contains(&e) && gbest.contains(&e))
                .copied()
                .collect();

            let beta: f64 = c4 * rng.random::<f64>();
            let n_elements: usize = number_of_elements(&mut rng, beta, &consensus_aisles);
            removal_of_elements(&mut rng, &consensus_aisles, n_elements, &mut velocity);

            for &(op, aisle) in &velocity {
                match op {
                    Op::Add => swarm[i].insert_aisle(aisle, problem),
                    Op::Remove => swarm[i].remove_aisle(aisle, problem)
                }
            }

            swarm[i].objective = problem.objective_function(
                problem.add_orders(&swarm[i].aisles_items),
                swarm[i].x.len() as u32
            );
        }
    }

    Ok(SolverOutput {
        convergence,
        time: start.elapsed().as_secs_f64(),
    })
}