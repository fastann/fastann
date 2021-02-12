mod bench;
mod bforest;
mod core;
mod flat;
mod hnsw;

fn main() {
    bench::bench::run_word_emb_demo();
}
