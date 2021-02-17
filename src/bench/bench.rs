// use crate::annoy;
// use crate::annoy::annoy::AnnoyIndexer;
use crate::bf;
use crate::bpforest;
use crate::core;
use crate::core::ann_index::ANNIndex;
use crate::core::parameters;
use crate::hnsw;
use crate::pq;
use pq::pq::PQIndexer;
use rand::distributions::{Alphanumeric, StandardNormal, Uniform};
use rand::distributions::{Distribution, Normal};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use hashbrown::HashMap;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

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

fn make_baseline(embs: Vec<Vec<f64>>, bf_idx: &mut Box<bf::bf::BruteForceIndex<f64, usize>>) {
    for i in 0..embs.len() {
        bf_idx.add_node(&core::node::Node::<f64, usize>::new_with_idx(&embs[i], i));
    }
    bf_idx.construct(core::metrics::Metric::CosineSimilarity);
}

fn make_bp_forest_baseline(
    embs: Vec<Vec<f64>>,
    bpforest_idx: &mut Box<bpforest::bpforest::BinaryProjectionForestIndex<f64, usize>>,
) {
    for i in 0..embs.len() {
        bpforest_idx.add_node(&core::node::Node::<f64, usize>::new_with_idx(&embs[i], i));
    }
    bpforest_idx.construct(core::metrics::Metric::CosineSimilarity);
}

fn make_baseline_for_word_emb(
    embs: &HashMap<String, Vec<f64>>,
    bf_idx: &mut bf::bf::BruteForceIndex<f64, String>,
) {
    for (key, value) in embs {
        bf_idx.add_node(&core::node::Node::<f64, String>::new_with_idx(
            &value,
            key.to_string(),
        ));
    }
    bf_idx.construct(core::metrics::Metric::CosineSimilarity);
}

// run for normal distribution test data
pub fn run_demo() {
    let (base, ns, ts) = make_normal_distribution_clustering(5, 1000, 1, 2, 100.0);
    let mut bf_idx = Box::new(bf::bf::BruteForceIndex::<f64, usize>::new(
        parameters::Parameters::default(),
    ));
    make_baseline(ns, &mut bf_idx);
    for i in ts.iter() {
        let result = bf_idx.search_k(i, 5);
        for j in result.iter() {
            println!("test base: {:?} neighbor: {:?}", i, j);
        }
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
            if idx == 500 {
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

    let mut bf_idx = Box::new(bf::bf::BruteForceIndex::<f64, usize>::new(
        parameters::Parameters::default(),
    ));
    make_baseline(train_data.clone(), &mut bf_idx);
    let mut bpforest_idx =
        Box::new(bpforest::bpforest::BinaryProjectionForestIndex::<f64, usize>::new(50, 6, -1));
    make_bp_forest_baseline(train_data.clone(), &mut bpforest_idx);
    // bpforest_idx.show_trees();
    let mut hnsw_idx = Box::new(hnsw::hnsw::HnswIndex::<f64, usize>::new(
        50,
        10000000,
        16,
        32,
        20,
        core::metrics::Metric::CosineSimilarity,
        40,
        false,
    ));
    make_hnsw_baseline(train_data.clone(), &mut hnsw_idx);

    let mut pq_idx=Box::new(pq::pq::PQIndexer::<f64, usize>::new(
        50,
        10,
        4,
        100,
        core::metrics::Metric::Euclidean,
    ));
    make_pq_baseline(train_data.clone(), &mut pq_idx);

    let indices: Vec<Box<ANNIndex<f64, usize>>> = vec![bf_idx, bpforest_idx, hnsw_idx, pq_idx];

    const K: i32 = 10;
    for i in 0..K {
        let mut rng = rand::thread_rng();

        let target_word: usize = rng.gen_range(1, words_vec.len());
        let w = words.get(&words_vec[target_word]).unwrap();

        for idx in indices.iter() {
            let start = SystemTime::now();
            let mut result = idx.search_k(&train_data[*w as usize], 20);
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
                let mut result = idx.search_k(&train_data[*w as usize], 20);
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

fn make_hnsw_baseline(embs: Vec<Vec<f64>>, hnsw_idx: &mut Box<hnsw::hnsw::HnswIndex<f64, usize>>) {
    for i in 0..embs.len() {
        // println!("addword i {:?}", i);
        hnsw_idx.add_node(&core::node::Node::<f64, usize>::new_with_idx(&embs[i], i));
    }
    println!("addword len {:?}", embs.len());
    // bpforest_idx.construct(core::metrics::Metric::CosineSimilarity);
}

fn make_pq_baseline(embs: Vec<Vec<f64>>, pq_idx: &mut Box<pq::pq::PQIndexer<f64, usize>>) {
    for i in 0..embs.len() {
        // println!("addword i {:?}", i);
        pq_idx.add_node(&core::node::Node::<f64, usize>::new_with_idx(&embs[i], i));
    }
    pq_idx.construct(core::metrics::Metric::Euclidean);
    println!("addword len {:?}", embs.len());
    // bpforest_idx.construct(core::metrics::Metric::CosineSimilarity);
}