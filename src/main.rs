mod annoy;
mod common;
mod hnsw;
use crate::annoy::annoy::AnnoyIndexer;
use rand::distributions::{Alphanumeric, StandardNormal, Uniform};
use rand::distributions::{Distribution, Normal};
use rand::{thread_rng, Rng};

fn test_annoy() {
    let a = vec![1.2, 2.3, 3.0];
    let b = vec![4.1, 5.0, 6.5];

    match common::calc::dot(&a, &b) {
        Ok(x) => println!("{}", x),
        Err(e) => println!("{}", e),
    }

    match common::calc::manhanttan_distance(&a, &b) {
        Ok(x) => println!("{}", x),
        Err(e) => println!("{}", e),
    }

    match common::calc::enclidean_distance(&a, &b) {
        Ok(x) => println!("{}", x),
        Err(e) => println!("{}", e),
    }

    println!("{}", common::calc::get_norm(&a));

    let angular = annoy::annoy::Angular {};
    let mut aix = annoy::annoy::AnnoyIndex::new(2, angular);

    let x: Vec<f64> = vec![1.1, 1.2, 60.6, 77.7, 88.8, 1.3, 1.4, 61.6, 78.7, 89.8];
    let y: Vec<f64> = vec![1.1, 1.2, 60.6, 77.7, 88.8, 1.3, 1.4, 61.6, 78.7, 89.8];

    for i in 0..x.len() {
        let f = vec![x[i], y[i]];
        aix.add_item(i as i32, &f, angular);
    }

    println!("{:?}", aix.build(1));
    println!("{:?}", aix.get_leaf(2));
    let f = vec![1.0, 1.0];
    println!("{:?}", aix);
    for i in 0..aix.leaves.len() {
        println!("{:?} {:?}", i, aix.leaves[i]);
    }
    println!("{:?}", aix.get_all_nns(&f, 2, 2));
}

fn test_hnsw() {
    let mut indexer = hnsw::hnsw::HnswIndexer::new(2);
    for i in 1..10 {
        let f = vec![i as f64, i as f64];
        indexer.add_item(i, &f);
    }
    let ret = indexer.search_knn(&vec![0.0, 0.0], 3).unwrap();
    for neigh in ret {
        println!("{:?}  {:?}", neigh._idx, neigh._distance);
    }
}

fn make_normal_distribution_clustering(
    clustering_n: usize,
    node_n: usize,
    dimension: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();

    let mut bases: Vec<Vec<f64>> = Vec::new();
    let mut ns: Vec<Vec<f64>> = Vec::new();
    for i in 0..clustering_n {
        let mut base: Vec<f64> = Vec::new();
        for i in 0..dimension {
            let n: f64 = rng.gen_range(0.0, 100.0); // base number
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

fn make_test_data() {
    let dimension = 2;
    let nodes = 5;
    let cluster = 3;

    let (base, vectors) = make_normal_distribution_clustering(cluster, nodes, dimension);
    for i in 0..vectors.len() {
        if i % nodes == 0 {
            println!("base: {:?}", base[i / nodes]);
        }
        println!("{:?}", vectors[i]);
    }
}

fn main() {
    println!("hello world");

    make_test_data();

    // test_hnsw();
}
