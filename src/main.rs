mod bench;
mod bf;
mod bpforest;
mod core;
mod hnsw;
mod mrng;
mod pq;





fn main() {
    // bench::bench::run_word_emb_demo();
    // bench::bench::run();
    // let guard = pprof::ProfilerGuard::new(100).unwrap();
    // let mut s = 0;
    // for iter in (0..10000) {
    //     s += iter;
    // }
    //         if let Ok(report) = guard.report().build() {
    //         let file = File::create(format!("flamegraph.svg")).unwrap();
    //         report.flamegraph(file).unwrap();
    //     };
    //     println!("s {:?}", s);
    bench::ann_bench::ann_bench();
}
