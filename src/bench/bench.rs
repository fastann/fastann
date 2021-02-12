// use crate::annoy;
// use crate::annoy::annoy::AnnoyIndexer;
use crate::core;
use crate::core::ann_index::AnnIndex;
use crate::core::parameters;
use crate::flat;
use crate::hnsw;
use rand::distributions::{Alphanumeric, StandardNormal, Uniform};
use rand::distributions::{Distribution, Normal};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

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

fn make_baseline(embs: Vec<Vec<f64>>, flat_idx: &mut flat::flat::FlatIndex<f64, usize>) {
    for i in 0..embs.len() {
        flat_idx.add(&core::node::Node::<f64, usize>::new_with_idx(&embs[i], i));
    }
    flat_idx.construct();
}

fn make_baseline_for_word_emb(
    embs: &HashMap<String, Vec<f64>>,
    flat_idx: &mut flat::flat::FlatIndex<f64, String>,
) {
    for (key, value) in embs {
        flat_idx.add(&core::node::Node::<f64, String>::new_with_idx(
            &value,
            key.to_string(),
        ));
    }
    flat_idx.construct();
}

// fn make_annoy_idx(embs: Vec<Vec<f64>>, aix: &mut annoy::annoy::AnnoyIndex<f64, annoy::annoy::Angular>) -> String {

//     let (base, mut vectors) = make_test_data();
//     vectors.shuffle(&mut thread_rng());
//     for i in 0..vectors.len() {
//         let f = vec![vectors[i][0], vectors[i][1]];
//         aix.add_item(i as i32, &f, angular);
//     }

//     println!("{:?}", aix.build(1));
//     aix.show_trees();
//     let f = &base[0];
//     for i in 0..aix.leaves.len() {
//         println!("{:?} {:?}", i, aix.leaves[i]);
//     }
//     println!("{:?}", aix.get_all_nns(&f, 2, 2));

//     let mut result_vec = Vec::new();
//     for leaf in aix.leaves {
//         result_vec.push(leaf.get_literal());
//     }
//     format!("[{}]", result_vec.join(","))
// }

// run for normal distribution test data
pub fn run_demo() {
    let (base, ns, ts) = make_normal_distribution_clustering(5, 1000, 1, 2, 100.0);
    let mut flat_idx = flat::flat::FlatIndex::<f64, usize>::new(parameters::Parameters::default());
    make_baseline(ns, &mut flat_idx);
    for i in ts.iter() {
        let result = flat_idx.search(i, 5, core::metrics::MetricType::CosineSimilarity);
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
    if let Ok(lines) = read_lines("src/bench/glove.6B.50d.txt") {
        let mut idx = 0;
        for line in lines {
            if let Ok(l) = line {
                let split_line = l.split(" ").collect::<Vec<&str>>();
                let word = split_line[0];
                let mut vecs = Vec::new();
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
    }

    let mut flat_idx = flat::flat::FlatIndex::<f64, usize>::new(parameters::Parameters::default());
    make_baseline(train_data.clone(), &mut flat_idx);

    // annoy
    // let angular = annoy::annoy::Angular::new();
    // let mut aix = annoy::annoy::AnnoyIndex::new(2, angular);

    const K: i32 = 10;
    for i in 0..K {
        let mut rng = rand::thread_rng();

        let target_word: usize = rng.gen_range(1, words_vec.len());
        let w = words.get(&words_vec[target_word]).unwrap();

        let result = flat_idx.search(&train_data[*w as usize], 20, core::metrics::MetricType::Dot);
        for (n, d) in result.iter() {
            println!(
                "target word: {}, neighbor: {:?}, distance: {:?}",
                words_vec[target_word],
                words_vec[n.idx().unwrap()],
                d
            );
        }
    }

    let test_words = vec![
        "frog", "china", "english", "football", "school", "computer", "apple", "math",
    ];
    for tw in test_words.iter() {
        if let Some(w) = words.get(&tw.to_string()) {
            let result = flat_idx.search(
                &train_data[*w as usize],
                20,
                core::metrics::MetricType::CosineSimilarity,
            );
            for (n, d) in result.iter() {
                println!(
                    "target word: {}, neighbor: {:?}, distance: {:?}",
                    tw,
                    words_vec[n.idx().unwrap()],
                    d
                );
            }
        }
    }
}
