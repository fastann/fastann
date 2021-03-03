// use crate::annoy;
// use crate::annoy::annoy::AnnoyIndexer;
use crate::bf;
use crate::bpforest;
use crate::core;
use crate::core::ann_index::ANNIndex;
use crate::hnsw;
use crate::pq;
use hashbrown::HashMap;
use pq::pq::PQIndex;
use prgrs::{writeln, Length, Prgrs};
use rand::distributions::{Alphanumeric, StandardNormal, Uniform};
use rand::distributions::{Distribution, Normal};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn make_normal_distribution_clustering(
    clustering_n: usize,
    node_n: usize,
    dimension: usize,
    range: f64,
) -> (
    Vec<Vec<f64>>, // center of cluster
    Vec<Vec<f64>>, // cluster data
) {
    let mut rng = rand::thread_rng();

    let mut bases: Vec<Vec<f64>> = Vec::new();
    let mut ns: Vec<Vec<f64>> = Vec::new();
    let normal = Normal::new(0.0, (range / 50.0));
    for i in 0..clustering_n {
        let mut base: Vec<f64> = Vec::with_capacity(dimension);
        for i in 0..dimension {
            let n: f64 = rng.gen_range(-range, range); // base number
            base.push(n);
        }

        for i in 0..node_n {
            let v_iter: Vec<f64> = rng.sample_iter(&normal).take(dimension).collect();
            let mut vec_item = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let vv = v_iter[i] + base[i]; // add normal distribution noise
                vec_item.push(vv);
            }
            // println!("{:?}", vec_item);
            ns.push(vec_item);
        }
        bases.push(base);
    }

    return (bases, ns);
}

// run for normal distribution test data
pub fn run_similarity_profile(test_time: usize) {
    let dimension = 50;
    let nodes_every_cluster = 20;
    let node_n = 4000;

    let (_, ns) =
        make_normal_distribution_clustering(node_n, nodes_every_cluster, dimension, 10000000.0);
    let mut bf_idx = Box::new(bf::bf::BruteForceIndex::<f64, usize>::new());
    let mut bpforest_idx = Box::new(
        bpforest::bpforest::BinaryProjectionForestIndex::<f64, usize>::new(dimension, 6, -1),
    );
    let mut hnsw_idx = Box::new(hnsw::hnsw::HnswIndex::<f64, usize>::new(
        dimension,
        100000,
        16,
        32,
        20,
        500,
        false,
    ));

    let mut pq_idx = Box::new(pq::pq::PQIndex::<f64, usize>::new(
        dimension,
        dimension / 2,
        4,
        100,
        core::metrics::Metric::Manhattan,
    ));

    // let mut indices: Vec<Box<ANNIndex<f64, usize>>> = vec![bpforest_idx];
    let mut indices: Vec<Box<ANNIndex<f64, usize>>> = vec![hnsw_idx];
    let mut accuracy = Arc::new(Mutex::new(Vec::new()));
    let mut cost = Arc::new(Mutex::new(Vec::new()));
    let mut base_cost = Arc::new(Mutex::new(Duration::default()));
    for i in 0..indices.len() {
        make_idx_baseline(ns.clone(), &mut indices[i]);
        accuracy.lock().unwrap().push(0.);
        cost.lock().unwrap().push(Duration::default());
    }
    make_idx_baseline(ns.clone(), &mut bf_idx);

    for i in Prgrs::new(0..test_time, 1000).set_length_move(Length::Proportional(0.5)) {
        // (0..test_time).into_par_iter().for_each(|_| {
        let mut rng = rand::thread_rng();

        let target: usize = rng.gen_range(0, ns.len());
        let w = ns.get(target).unwrap();

        let base_start = SystemTime::now();
        let base_result = bf_idx.search_k(&w, 100);
        let mut base_set = HashSet::new();
        for (n, _) in base_result.iter() {
            base_set.insert(n.idx().unwrap().clone());
        }
        let base_since_the_epoch = SystemTime::now()
            .duration_since(base_start)
            .expect("Time went backwards");
        *base_cost.lock().unwrap() += base_since_the_epoch;

        for j in 0..indices.len() {
            let start = SystemTime::now();
            let result = indices[j].search_k(&w, 100);
            for (n, _) in result.iter() {
                if base_set.contains(&n.idx().unwrap()) {
                    accuracy.lock().unwrap()[j] += 1.0;
                }
            }
            let since_the_epoch = SystemTime::now()
                .duration_since(start)
                .expect("Time went backwards");
            cost.lock().unwrap()[j] += since_the_epoch;
        }
    }
    // });

    println!(
        "test for {:?} times, nodes {:?}, base use {:?} millisecond",
        test_time,
        nodes_every_cluster * node_n,
        base_cost.lock().unwrap().as_millis() as f64 / (test_time as f64)
    );
    for i in 0..indices.len() {
        let a = accuracy.lock().unwrap()[i];
        println!(
            "index: {:?}, avg accuracy: {:?}, hit {:?}, avg cost {:?} millisecond",
            indices[i].name(),
            a / (test_time as f64),
            a,
            cost.lock().unwrap()[i].as_millis() as f64 / (test_time as f64),
        );
    }
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

// run for exist embedding file
pub fn run_word_emb_demo() {
    let mut words = HashMap::new();
    let mut word_idxs = HashMap::new();
    let mut words_vec = Vec::new();
    let mut train_data = Vec::new();
    let mut words_train_data = HashMap::new();
    let file = File::open("src/bench/glove.6B.50d.txt").unwrap();
    let reader = BufReader::new(file);

    let mut idx = 0;
    for line in reader.lines() {
        if let Ok(l) = line {
            if idx == 80000 {
                break;
            }
            let split_line = l.split(" ").collect::<Vec<&str>>();
            let word = split_line[0];
            let mut vecs = Vec::with_capacity(split_line.len() - 1);
            for i in 1..split_line.len() {
                vecs.push(split_line[i].parse::<f64>().unwrap());
            }
            words.insert(word.to_string(), idx.clone());
            word_idxs.insert(idx.clone(), word.to_string());
            words_vec.push(word.to_string());
            words_train_data.insert(word.to_string(), vecs.clone());
            idx += 1;
            train_data.push(vecs.clone());
            if (idx % 100000 == 0) {
                println!("load {:?}", idx);
            }
        }
    }

    let mut bf_idx = Box::new(bf::bf::BruteForceIndex::<f64, usize>::new());
    let mut bpforest_idx =
        Box::new(bpforest::bpforest::BinaryProjectionForestIndex::<f64, usize>::new(50, 6, -1));
    // bpforest_idx.show_trees();
    let mut hnsw_idx = Box::new(hnsw::hnsw::HnswIndex::<f64, usize>::new(
        50,
        10000000,
        16,
        32,
        20,
        500,
        false,
    ));

    let mut pq_idx = Box::new(pq::pq::PQIndex::<f64, usize>::new(
        50,
        10,
        4,
        100,
        core::metrics::Metric::Manhattan,
    ));

    // let indices: Vec<Box<ANNIndex<f64, usize>>> = vec![bf_idx, bpforest_idx, hnsw_idx, pq_idx];
    let mut indices: Vec<Box<ANNIndex<f64, usize>>> = vec![bf_idx, bpforest_idx];
    for i in 0..indices.len() {
        make_idx_baseline(train_data.clone(), &mut indices[i]);
    }

    const K: i32 = 10;
    for i in 0..K {
        let mut rng = rand::thread_rng();

        let target_word: usize = rng.gen_range(1, words_vec.len());
        let w = words.get(&words_vec[target_word]).unwrap();

        for idx in indices.iter() {
            let start = SystemTime::now();
            let mut result = idx.search_k(&train_data[*w as usize], 10);
            for (n, d) in result.iter() {
                println!(
                    "{:?} target word: {}, neighbor: {:?}, distance: {:?}",
                    idx.name(),
                    words_vec[target_word],
                    words_vec[n.idx().unwrap()],
                    d
                );
            }
            let since_the_epoch = SystemTime::now()
                .duration_since(start)
                .expect("Time went backwards");
            println!("{:?}: {:?}", idx.name(), since_the_epoch);
        }
    }

    let test_words = vec![
        "frog", "china", "english", "football", "school", "computer", "apple", "math",
    ];
    for tw in test_words.iter() {
        if let Some(w) = words.get(&tw.to_string()) {
            for idx in indices.iter() {
                let start = SystemTime::now();
                let mut result = idx.search_k(&train_data[*w as usize], 10);
                for (n, d) in result.iter() {
                    println!(
                        "{:?} target word: {}, neighbor: {:?}, distance: {:?}",
                        idx.name(),
                        tw,
                        words_vec[n.idx().unwrap()],
                        d
                    );
                }
                let since_the_epoch = SystemTime::now()
                    .duration_since(start)
                    .expect("Time went backwards");
                println!("{:?}: {:?}", idx.name(), since_the_epoch);
            }
        }
    }
}

fn make_idx_baseline<T: ANNIndex<f64, usize> + ?Sized>(embs: Vec<Vec<f64>>, idx: &mut Box<T>) {
    for i in 0..embs.len() {
        idx.add_node(&core::node::Node::<f64, usize>::new_with_idx(&embs[i], i));
    }
    idx.construct(core::metrics::Metric::Manhattan).unwrap();
}
