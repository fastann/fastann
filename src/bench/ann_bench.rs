#![deny(clippy::all)]
use crate::core;
use crate::core::ann_index::ANNIndex;

use crate::hnsw;
use crate::mrng;
use crate::pq;

use std::time::SystemTime;
use std::{collections::HashSet, u128};

struct StatMetrics {
    QPS: f64,
    Accuracy: usize,
    Cost: f64,
    BuildCost: f64,
    TestSize: usize,
}

const data_path: &str =
    "/Users/chenyangyang/rust/fastann/src/bench/fashion-mnist-784-euclidean.hdf5";
const dimension: usize = 784;
const K: usize = 10;

pub fn ann_bench() {
    let file = hdf5::File::open(&data_path).unwrap();
    let train: Vec<Vec<f32>> = file
        .dataset("train")
        .unwrap()
        .read_raw::<f32>()
        .unwrap()
        .chunks(dimension)
        .map(|s| s.to_vec())
        .collect();
    let test: Vec<Vec<f32>> = file
        .dataset("test")
        .unwrap()
        .read_raw::<f32>()
        .unwrap()
        .chunks(dimension)
        .map(|s| s.to_vec())
        .collect();
    let neighbors: Vec<HashSet<usize>> = file
        .dataset("neighbors")
        .unwrap()
        .read_raw::<usize>()
        .unwrap()
        .chunks(100)
        .map(|s| s[..K].iter().cloned().collect::<HashSet<usize>>())
        .collect();

    // let _bf_idx = Box::new(bf::bf::BruteForceIndex::<f32, usize>::new());
    // let mut bpforest_idx = Box::new(
    //     bpforest::bpforest::BinaryProjectionForestIndex::<f32, usize>::new(dimension, 6, -1),
    // );
    // let _hnsw_idx = Box::new(hnsw::hnsw::HNSWIndex::<f32, usize>::new(
    //     dimension, 100000, 16, 32, 20, 500, false,
    // ));

    // let mut pq_idx = Box::new(pq::pq::PQIndex::<f32, usize>::new(
    //     dimension,
    //     dimension / 2,
    //     4,
    //     100,
    // ));
    // let mut ssg_idx = Box::new(mrng::ssg::SatelliteSystemGraphIndex::<f32, usize>::new(
    //     dimension,
    //     &mrng::ssg::SatelliteSystemGraphParams::default(),
    // ));

    // make_idx_baseline(train.clone(), &mut bf_idx);
    // make_idx_baseline(train.clone(), &mut bpforest_idx);
    // make_idx_baseline(train.clone(), &mut hnsw_idx);
    // make_idx_baseline(train.clone(), &mut pq_idx);
    // make_idx_baseline(train, &mut ssg_idx);
    println!("train len: {:?}", train.len());
    println!("test len: {:?}", test.len());
    // bench_hnsw(&train, &test, &neighbors);
    bench_ssg(&train, &test, &neighbors);
    // bench_ivfpq(&train, &test, &neighbors);
}

fn bench_ssg<E: core::node::FloatElement>(
    train: &Vec<Vec<E>>,
    test: &Vec<Vec<E>>,
    neighbors: &Vec<HashSet<usize>>,
) {
    let params_set = vec![
        mrng::ssg::SatelliteSystemGraphParams::<E>::default()
            .angle(60.0)
            .init_k(20)
            .index_size(20)
            .neighbor_neighbor_size(30)
            .root_size(20),
        // mrng::ssg::SatelliteSystemGraphParams::default()
        //     .angle(60.0)
        //     .init_k(50)
        //     .index_size(50)
        //     .neighbor_neighbor_size(50)
        //     .root_size(20),
        // mrng::ssg::SatelliteSystemGraphParams::default()
        //     .angle(60.0)
        //     .init_k(50)
        //     .index_size(50)
        //     .neighbor_neighbor_size(50)
        //     .root_size(50),
    ];

    let mut metrics_stats: Vec<StatMetrics> = Vec::new();
    for params in params_set.iter() {
        println!("start params {:?}", params);
        let mut ssg_idx = Box::new(mrng::ssg::SatelliteSystemGraphIndex::<E, usize>::new(
            dimension, params,
        ));
        make_idx_baseline(train, &mut ssg_idx);
        metrics_stats.push(bench_calc(ssg_idx, test, neighbors));
        println!("finish params {:?}", params);
    }

    for i in 0..metrics_stats.len() {
        println!(
            "idx ssg params {:?} result {:?}/{:?} {:?}ms qps {:?}",
            params_set[i],
            metrics_stats[i].Accuracy,
            metrics_stats[i].TestSize,
            metrics_stats[i].Cost,
            metrics_stats[i].QPS,
        );
    }
}

fn bench_hnsw<E: core::node::FloatElement>(
    train: &Vec<Vec<E>>,
    test: &Vec<Vec<E>>,
    neighbors: &Vec<HashSet<usize>>,
) {
    let params_set = vec![
        hnsw::hnsw::HNSWParams::<E>::default()
            .max_item(100000)
            .n_neighbor(16)
            .n_neighbor0(32)
            .ef_build(500)
            .ef_search(16)
            .has_deletion(false),
        hnsw::hnsw::HNSWParams::<E>::default()
            .max_item(100000)
            .n_neighbor(8)
            .n_neighbor0(16)
            .ef_build(500)
            .ef_search(16)
            .has_deletion(false),
        hnsw::hnsw::HNSWParams::<E>::default()
            .max_item(100000)
            .n_neighbor(16)
            .n_neighbor0(32)
            .ef_build(500)
            .ef_search(16)
            .has_deletion(false),
    ];

    let mut metrics_stats: Vec<StatMetrics> = Vec::new();
    for params in params_set.iter() {
        let mut hnsw_idx = Box::new(hnsw::hnsw::HNSWIndex::<E, usize>::new(dimension, params));
        make_idx_baseline(train, &mut hnsw_idx);
        metrics_stats.push(bench_calc(hnsw_idx, test, neighbors));
        println!("finish params {:?}", params);
    }

    for i in 0..metrics_stats.len() {
        println!(
            "idx hnsw params {:?} result {:?}/{:?} {:?}ms qps {:?}",
            params_set[i],
            metrics_stats[i].Accuracy,
            metrics_stats[i].TestSize,
            metrics_stats[i].Cost,
            metrics_stats[i].QPS,
        );
    }
}

fn bench_ivfpq<E: core::node::FloatElement>(
    train: &Vec<Vec<E>>,
    test: &Vec<Vec<E>>,
    neighbors: &Vec<HashSet<usize>>,
) {
    let params_set = vec![pq::pq::IVFPQParams::<E>::default()
        .n_sub(16)
        .sub_bits(4)
        .n_kmeans_center(256)
        .search_n_center(4)
        .train_epoch(100)];

    let mut metrics_stats: Vec<StatMetrics> = Vec::new();
    for params in params_set.iter() {
        let mut ivfpq_idx = Box::new(pq::pq::IVFPQIndex::<E, usize>::new(dimension, params));
        make_idx_baseline(train, &mut ivfpq_idx);
        metrics_stats.push(bench_calc(ivfpq_idx, test, neighbors));
        println!("finish params {:?}", params);
    }

    for i in 0..metrics_stats.len() {
        println!(
            "idx ivfpq params {:?} result {:?}/{:?} {:?}ms qps {:?}",
            params_set[i],
            metrics_stats[i].Accuracy,
            metrics_stats[i].TestSize,
            metrics_stats[i].Cost,
            metrics_stats[i].QPS,
        );
    }
}

fn bench_calc<E: core::node::FloatElement, T: ANNIndex<E, usize> + ?Sized>(
    ann_idx: Box<T>,
    test: &Vec<Vec<E>>,
    neighbors: &Vec<HashSet<usize>>,
) -> StatMetrics {
    let mut accuracy = 0;
    let mut cost = 0.0;
    for idx in 0..test.len() {
        let start = SystemTime::now();
        let result = ann_idx.search_k_ids(test[idx].as_slice(), K);
        let since_start = SystemTime::now().duration_since(start).expect("error");
        cost += (since_start.as_micros() as f64) / 1000.0;
        let true_set = &neighbors[idx];
        result.iter().for_each(|candidate| {
            if true_set.contains(candidate) {
                accuracy += 1;
            }
        });
    }
    println!("cost: {:?}", cost);
    println!(
        "{:?} result {:?}/{:?} {:?}ms qps {:?}",
        ann_idx.name(),
        accuracy,
        test.len() * K,
        cost,
        1.0 / (((cost as f32) / 1000.0) / test.len() as f32)
    );
    StatMetrics {
        QPS: 1.0 / (((cost as f64) / 1000.0) / test.len() as f64),
        Accuracy: accuracy,
        TestSize: test.len() * K,
        Cost: cost,
        BuildCost: 0.0,
    }
}

fn make_idx_baseline<E: core::node::FloatElement, T: ANNIndex<E, usize> + ?Sized>(
    embs: &Vec<Vec<E>>,
    idx: &mut Box<T>,
) {
    let start = SystemTime::now();
    for i in 0..embs.len() {
        // println!("{:?}", embs[i].len());
        idx.add_node(&core::node::Node::<E, usize>::new_with_idx(
            embs[i].as_slice(),
            i,
        ))
        .unwrap();
    }
    idx.construct(core::metrics::Metric::Euclidean).unwrap();
    let since_start = SystemTime::now()
        .duration_since(start)
        .expect("Time went backwards");

    println!(
        "index {:?} build time {:?} ms",
        idx.name(),
        since_start.as_millis() as f64
    );
}
