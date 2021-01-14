use crate::annoy;
use crate::annoy::annoy::AnnoyIndexer;
use crate::common;
use crate::hnsw;
use rand::distributions::{Alphanumeric, StandardNormal, Uniform};
use rand::distributions::{Distribution, Normal};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

fn test_distance_calc() {
    let a = vec![1.2, 2.3, 3.0];
    let b = vec![4.1, 5.0, 6.5];

    match common::calc::dot(&a, &b) {
        Ok(x) => println!("{}", x),
        Err(e) => println!("{}", e),
    }

    match common::calc::manhattan_distance(&a, &b) {
        Ok(x) => println!("{}", x),
        Err(e) => println!("{}", e),
    }

    match common::calc::euclidean_distance(&a, &b) {
        Ok(x) => println!("{}", x),
        Err(e) => println!("{}", e),
    }

    println!("{}", common::calc::get_norm(&a));
}

fn test_annoy() -> String {
    let angular = annoy::annoy::Angular::new();
    let mut aix = annoy::annoy::AnnoyIndex::new(2, angular);

    let (base, mut vectors) = make_test_data();
    vectors.shuffle(&mut thread_rng());
    for i in 0..vectors.len() {
        let f = vec![vectors[i][0], vectors[i][1]];
        aix.add_item(i as i32, &f, angular);
    }

    println!("{:?}", aix.build(1));
    aix.show_trees();
    let f = &base[0];
    for i in 0..aix.leaves.len() {
        println!("{:?} {:?}", i, aix.leaves[i]);
    }
    println!("{:?}", aix.get_all_nns(&f, 2, 2));

    let mut result_vec = Vec::new();
    for leaf in aix.leaves {
        result_vec.push(leaf.get_literal());
    }
    format!("[{}]", result_vec.join(","))
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

fn make_test_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let dimension = 2;
    let nodes = 50;
    let cluster = 3;

    let (base, vectors) = make_normal_distribution_clustering(cluster, nodes, dimension);
    for i in 0..vectors.len() {
        if i % nodes == 0 {
            println!("base: {:?}", base[i / nodes]);
        }
        println!("{:?}", vectors[i]);
    }
    return (base, vectors);
}

pub fn run_demo() -> String {
    println!("hello world");

    test_annoy()
}
