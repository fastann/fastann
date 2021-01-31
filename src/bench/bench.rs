use crate::annoy;
use crate::annoy::annoy::AnnoyIndexer;
use crate::core;
use crate::core::ann_index::AnnIndex;
use crate::core::parameters;
use crate::flat;
use crate::hnsw;
use rand::distributions::{Alphanumeric, StandardNormal, Uniform};
use rand::distributions::{Distribution, Normal};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

fn make_normal_distribution_clustering(
    clustering_n: usize,
    node_n: usize,
    test_n: usize,
    dimension: usize,
    range: f64,
) -> (
    Vec<Vec<f64>>, // center of cluster
    Vec<Vec<f64>>, // cluster data
    Vec<Vec<f64>>, // test data
) {
    let mut rng = rand::thread_rng();

    let mut bases: Vec<Vec<f64>> = Vec::new();
    let mut ns: Vec<Vec<f64>> = Vec::new();
    let mut ts: Vec<Vec<f64>> = Vec::new();
    for i in 0..clustering_n {
        let mut base: Vec<f64> = Vec::new();
        for i in 0..dimension {
            let n: f64 = rng.gen_range(0.0, range); // base number
            base.push(n);
        }

        for i in 0..node_n {
            let v_iter: Vec<f64> = rng.sample_iter(&StandardNormal).take(dimension).collect();
            let mut vec_item = Vec::new();
            for i in 0..dimension {
                let vv = v_iter[i] + base[i]; // add normal distribution noise
                vec_item.push(vv);
            }
            ns.push(vec_item);
        }

        for i in 0..test_n {
            let v_iter: Vec<f64> = rng.sample_iter(&StandardNormal).take(dimension).collect();
            let mut vec_item = Vec::new();
            for i in 0..dimension {
                let vv = v_iter[i] + base[i]; // add normal distribution noise
                vec_item.push(vv);
            }
            ts.push(vec_item);
        }
        bases.push(base);
    }

    return (bases, ns, ts);
}

fn make_baseline(embs: Vec<Vec<f64>>, flat_idx: &mut flat::flat::FlatIndex<f64>) {
    for i in 0..embs.len() {
        flat_idx.add(&core::node::Node::<f64>::new_with_id(&embs[i], i));
    }
    flat_idx.construct();
}

pub fn run_demo() {
    let (base, ns, ts) = make_normal_distribution_clustering(5, 1000, 1, 2, 100.0);
    let mut flat_idx = flat::flat::FlatIndex::<f64>::new(parameters::Parameters::default());
    make_baseline(ns, &mut flat_idx);
    for i in ts.iter() {
        let result = flat_idx.search(i, 5, &core::metrics::manhattan_distance);
        for j in result.iter() {
            println!("test base: {:?} neighbor: {:?}", i, j);
        }
    }
}
