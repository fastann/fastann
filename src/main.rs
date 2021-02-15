use fastann::bench::bench::{run_word_emb_demo, run_word_emb_hnsw_demo};

mod bench;
mod bpforest;
mod hnsw;
mod core;
mod flat;

fn main() {
    // bench::bench::run_word_emb_demo();
    bench::bench::run_word_emb_hnsw_demo();
}
