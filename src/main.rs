  
mod bench;
mod bf;
mod bpforest;
mod core;
mod hnsw;
mod mrng;
mod pq;

fn main() {
    // bench::bench::run_word_emb_demo();
    bench::bench::run();
}
