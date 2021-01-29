use crate::annoy;
use crate::annoy::annoy::AnnoyIndexer;
use crate::common;
use crate::hnsw;
use crate::flat;
use rand::distributions::{Alphanumeric, StandardNormal, Uniform};
use rand::distributions::{Distribution, Normal};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

fn make_normal_distribution_clustering(
    clustering_n: usize,
    node_n: usize,
    dimension: usize,
    range: f64
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();

    let mut bases: Vec<Vec<f64>> = Vec::new();
    let mut ns: Vec<Vec<f64>> = Vec::new();
    for i in 0..clustering_n {
        let mut base: Vec<f64> = Vec::new();
        for i in 0..dimension {
            let n: f64 = rng.gen_range(0.0, range); // base number
            base.push(n);
        }

        let mut v: Vec<f64> = Vec::new();
        for i in 0..node_n {
            let v_iter: Vec<f64> = rng.sample_iter(&StandardNormal).take(dimension).collect();
            let mut vec_item = Vec::new();
            for i in 0..dimension {
                let vv = v_iter[i] + base[i]; // add normal distribution noise
                vec_item.push(vv);
            }
            ns.push(vec_item);
        }
        bases.push(base);
    }

    return (bases, ns);
}

fn baseline(emb: Vec<Vec<f64>>,) {
    flat_idx = FlatIdx::new();
    for i in emb.iter() {
        flat_idx.add(node::Node<f64>::new(i));
    }
    flat_idx.train();
}

pub fn run_demo() -> String {

    let (test_data, ns) = make_normal_distribution_clustering(5, 1000, 20, 100.0);
    baseline(test_data);
}
