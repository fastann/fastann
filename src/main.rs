mod bench;
mod bf;
mod bpforest;
mod core;
mod hnsw;
mod pq;

fn main() {
    bench::bench::run_word_emb_demo();
}
