mod annoy;
mod bench;
mod core;
mod flat;
mod hnsw;

fn main() {
    bench::bench::run_demo();
}
