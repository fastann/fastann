  
mod bench;
mod bf;
mod bpforest;
mod core;
mod hnsw;
mod pq;

fn main() {
    bench::bench::run_similarity_profile(1000);
}
