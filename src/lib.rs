//! # Methods for Order Batching.
//! Main module for the order batching problem.
//!
//! Reads and structures the input data for use by the optimization method.

use anyhow::{Context, Error};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};

pub mod methods;
pub use methods::sbpso::sbpso;

/// Represents an instance of the order batching problem.
///
/// Loaded from a dataset file in the expected format, containing orders, aisles, and capacity limits.
pub struct Problem {
    o: usize,
    i: usize,
    a: usize,
    orders: Vec<HashMap<usize, u32>>,
    sorted_orders: Vec<(usize, u32)>,
    aisles: Vec<HashMap<usize, u32>>,
    lb: u32,
    ub: u32,
}

impl Problem {
    /// Create and populate a `Problem` using the values from the dataset.
    ///
    /// # Errors
    /// This functin will return an error if `path` doesn't exist, cannot be read, or is in an invalid format.
    ///
    /// # Examples
    /// ```
    /// use pso_for_order_batching_rust::Problem;
    ///
    /// fn main() -> anyhow::Result<()> {
    ///     let problem: Problem = Problem::new("instance.txt")?;
    ///     problem.print_elements();
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let file: File = File::open(path)?;
        let mut lines: Lines<BufReader<File>> = BufReader::new(file).lines();

        // First line: o i a.
        let first: String = next_line(&mut lines, "first line (o i a)")?;
        let mut iter: std::str::SplitWhitespace<'_> = first.split_whitespace();
        
        let o: usize = parse_next(&mut iter, "o")? as usize;
        let i: usize = parse_next(&mut iter, "i")? as usize;
        let a: usize = parse_next(&mut iter, "a")? as usize;

        // Next `o` lines: orders.
        let mut orders: Vec<HashMap<usize, u32>> = Vec::with_capacity(o);
        let mut sorted_orders: Vec<(usize, u32)> = Vec::with_capacity(o);
        
        for idx in 0..o {
            let line: String = next_line(&mut lines, &format!("order {idx} line"))?;
            let (k, pairs) = parse_map_line(&line)?;

            let mut order: HashMap<usize, u32> = HashMap::with_capacity(k);
            let mut sum: u32 = 0;
            for (item, qty) in pairs {
                order.insert(item, qty);
                sum += qty;
            }
            orders.push(order);
            sorted_orders.push((idx, sum));
        }

        sorted_orders.sort_unstable_by_key(|&(_, sum)| Reverse(sum));

        // Next `a` lines: aisles.
        let mut aisles: Vec<HashMap<usize, u32>> = Vec::with_capacity(a);

        for idx in 0..a {
            let line: String = next_line(&mut lines, &format!("aisle {idx} line"))?;
            let (k, pairs) = parse_map_line(&line)?;

            let mut aisle: HashMap<usize, u32> = HashMap::with_capacity(k);
            for (item, qty) in pairs {
                aisle.insert(item, qty);
            }
            aisles.push(aisle);
        }

        // Last line: lb ub.
        let last: String = next_line(&mut lines, "last line (lb ub)")?;
        let mut iter: std::str::SplitWhitespace<'_> = last.split_whitespace();

        let lb: u32 = parse_next(&mut iter, "lb")?;
        let ub: u32 = parse_next(&mut iter, "ub")?;

        Ok(Self {
            o,
            i,
            a,
            orders,
            sorted_orders,
            aisles,
            lb,
            ub,
        })
    }

    /// Greedily selects the orders with the largest number of items that do not violate capacity and supply constraints.
    /// 
    /// To ensure that the aisles always have the best combination of orders, this function assumes that no orders have 
    /// been selected at the time it is called. And for optimization purposes, this function returns only the quantity of 
    /// items in the selected orders.
    pub fn add_orders(
        &self,
        aisles_items: &Vec<u32>
    ) -> u32 {
        let mut number_items: u32 = 0;

        let mut available_items: Vec<u32> = aisles_items.clone();
        for &(order, qty) in &self.sorted_orders {
            if number_items + qty <= self.ub {
                let mut valid: bool = true;

                for (&item, &item_qty) in &self.orders[order] {
                    if item_qty > available_items[item] {
                        valid = false;
                        break;
                    }
                }

                if valid {
                    number_items += qty;
                    for (&item, &item_qty) in &self.orders[order] {
                        available_items[item] -= item_qty;
                    }
                }
            }
        }

        number_items
    }

    /// Calculates the value of the objective function and returns it. If the lower bound constraint is violated, or if `number_aisles` is 0, the return value will be 0.
    /// 
    /// Since the methods work directly with the aisles, which have no constraints, and use a greedy function that selects orders while respecting the upper bound and supply constraints (`add_orders`), this function does not check the previous constraints to avoid redundancy.
    pub fn objective_function(
        &self,
        number_items: u32,
        number_aisles: u32
    ) -> f64 {
        if number_items < self.lb || number_aisles == 0 {
            return 0.0;
        };

        number_items as f64 / number_aisles as f64
    }

    /// Prints all elements of the problem (for debugging).
    pub fn print_elements(&self) {
        println!("Dataset: o={} i={} a={}", self.o, self.i, self.a);

        println!("\nOrders:");
        for (idx, order) in self.orders.iter().enumerate() {
            print!("  [{idx}]: ");
            for (item, qty) in order {
                print!("item={item} qty={qty}  ");
            }
            println!();
        }

        println!("\nSorted orders:");
        for (idx, sum) in &self.sorted_orders {
            print!("  (pedido={idx}, total={sum}) ");
        }

        println!("\n\nAisles:");
        for (idx, aisle) in self.aisles.iter().enumerate() {
            print!("  [{idx}]: ");
            for (item, qty) in aisle {
                print!("item={item} qty={qty}  ");
            }
            println!();
        }

        println!("\nlb={} ub={}", self.lb, self.ub);
    }
}

/// Reads the next line from `lines`, returning a contextualized error if none is available.
fn next_line(
    lines: &mut impl Iterator<Item = io::Result<String>>,
    context: &str,
) -> anyhow::Result<String> {
    lines
        .next()
        .ok_or_else(|| Error::msg(format!("Missing {context}")))?
        .context("Something got wrong while reading the line")
}

/// Parses the next token from the `iter` as u32.
fn parse_next<'a>(
    iter: &mut impl Iterator<Item = &'a str>,
    field: &str
) -> anyhow::Result<u32> {
    iter.next()
        .ok_or_else(|| Error::msg(format!("Missing field: '{field}'")))?
        .parse()
        .with_context(|| format!("Something got wrong while parsing '{field}'"))
}

/// Parses the `line` in the following format: k item1 qty1 item2 qty2 ...
fn parse_map_line(
    line: &str
) -> anyhow::Result<(usize, Vec<(usize, u32)>)> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let k: usize = parts[0].parse().context("Something got wrong while parsing k")?;
    let pairs: Vec<(usize, u32)> = parse_pairs(&parts[1..])?;

    if pairs.len() != k {
        anyhow::bail!("Expected {k} pairs, got {}", pairs.len());
    }

    Ok((k, pairs))
}

/// Converts string slices into (usize, u32) pairs.
fn parse_pairs(
    parts: &[&str]
) -> anyhow::Result<Vec<(usize, u32)>> {
    parts
        .chunks(2)
        .map(|chunk| Ok((chunk[0].parse()?, chunk[1].parse()?)))
        .collect()
}