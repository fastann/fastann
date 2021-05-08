use crate::bf;
use crate::bpforest;
use crate::core;
use crate::core::ann_index::ANNIndex;
use crate::core::ann_index::SerializableIndex;
use crate::core::arguments;
use crate::hnsw;
use crate::mrng;
use crate::pq;
use std::time::{Duration, SystemTime};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

pub fn ann_bench() {
    let data_path =
        "/Users/chenyangyang/pkg/ann_bench/ann-benchmarks/data/fashion-mnist-784-euclidean.hdf5";
    let dimension = 784;
    let K = 10;

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

    let mut bf_idx = Box::new(bf::bf::BruteForceIndex::<f32, usize>::new());
    // let mut bpforest_idx = Box::new(
    //     bpforest::bpforest::BinaryProjectionForestIndex::<f32, usize>::new(dimension, 6, -1),
    // );
    let mut hnsw_idx = Box::new(hnsw::hnsw::HNSWIndex::<f32, usize>::new(
        dimension, 100000, 16, 32, 20, 500, false,
    ));

    // let mut pq_idx = Box::new(pq::pq::PQIndex::<f32, usize>::new(
    //     dimension,
    //     dimension / 2,
    //     4,
    //     100,
    // ));
    let mut ssg_idx = Box::new(mrng::ssg::SatelliteSystemGraphIndex::<f32, usize>::new(
        dimension, &mrng::ssg::SatelliteSystemGraphParams::default(),
    ));

    // make_idx_baseline(train.clone(), &mut bf_idx);
    // make_idx_baseline(train.clone(), &mut bpforest_idx);
    // make_idx_baseline(train.clone(), &mut hnsw_idx);
    // make_idx_baseline(train.clone(), &mut pq_idx);
    make_idx_baseline(train, &mut ssg_idx);

    {
        let ann_idx = &ssg_idx;
        let mut accuracy = 0;
        let mut cost = 0;
        for idx in 0..test.len() {
            let test_data = &test[idx];
            let start = SystemTime::now();
            let result = ann_idx.search_k_ids(test_data, K);
            let since_start = SystemTime::now().duration_since(start).expect("error");
            cost = cost + since_start.as_millis();
            let true_set = &neighbors[idx];
            result.iter().for_each(|candidate| {
                if true_set.contains(candidate) {
                    accuracy += 1;
                }
            });
            // println!("{:?}/{:?} true_set {:?} accuracy, , my ans {:?}", accuracy, true_set.len() * idx, true_set, result);
        }
        println!(
            "{:?} result {:?}/{:?} {:?}ms qps {:?}",
            ann_idx.name(),
            accuracy,
            test.len() * K,
            cost,
            1.0 / (((cost as f32) / 1000.0) / test.len() as f32)
        );
    }
}

fn make_idx_baseline<E: core::node::FloatElement, T: ANNIndex<E, usize> + ?Sized>(
    embs: Vec<Vec<E>>,
    idx: &mut Box<T>,
) {
    let start = SystemTime::now();
    for i in 0..embs.len() {
        // println!("{:?}", embs[i].len());
        idx.add_node(&core::node::Node::<E, usize>::new_with_idx(&embs[i], i))
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
