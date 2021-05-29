// use crate::annoy;
// use crate::annoy::annoy::AnnoyIndexer;
use fastann::bf;
use fastann::bpforest;
use fastann::core;
use fastann::core::ann_index::ANNIndex;
use fastann::core::ann_index::SerializableIndex;
use fastann::core::arguments;
use fastann::hnsw;
use fastann::mrng;
use fastann::pq;
#[cfg(feature = "without_std")]
use hashbrown::HashMap;

use prgrs::{Length, Prgrs};

use rand::distributions::{Distribution, Normal}; 

use rand::Rng;

use rayon::prelude::*;
#[cfg(not(feature = "without_std"))]
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
const LINE_SIZE: usize = 10000;
// rayon::ThreadPoolBuilder::new()
//     .num_threads(4)
//     .build_global()
//     .unwrap();
const DIMENSION: usize = 50;
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
    let normal = Normal::new(0.0, range / 50.0);
    for _i in 0..clustering_n {
        let mut base: Vec<f64> = Vec::with_capacity(dimension);
        for _i in 0..dimension {
            let n: f64 = rng.gen_range(-range..range); // base number
            base.push(n);
        }

        for _i in 0..node_n {
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

    (bases, ns)
}

// run for normal distribution test data
pub fn run_similarity_profile(test_time: usize) {
    let dimension = 2;
    let nodes_every_cluster = 3;
    let node_n = 5000;

    let (_, ns) =
        make_normal_distribution_clustering(node_n, nodes_every_cluster, dimension, 100.0);
    let mut bf_idx = Box::new(bf::bf::BruteForceIndex::<f64, usize>::new());
    let _bpforest_idx = Box::new(
        bpforest::bpforest::BinaryProjectionForestIndex::<f64, usize>::new(dimension, 6, -1),
    );
    let hnsw_idx = Box::new(hnsw::hnsw::HNSWIndex::<f64, usize>::new(
        dimension,
        &hnsw::hnsw::HNSWParams::default(),
    ));

    let _pq_idx = Box::new(pq::pq::PQIndex::<f64, usize>::new(
        dimension,
        &pq::pq::PQParams::<f64>::default()
        .n_sub(DIMENSION/2)
        .sub_bits(4)
        .train_epoch(100)
    ));
    let _ssg_idx = Box::new(mrng::ssg::SatelliteSystemGraphIndex::<f64, usize>::new(
        dimension,
        &mrng::ssg::SatelliteSystemGraphParams::default(),
    ));

    let mut indices: Vec<Box<ANNIndex<f64, usize>>> = vec![hnsw_idx];
    // let mut indices: Vec<Box<dyn ANNIndex<f64, usize>>> =
    //     vec![ssg_idx, bpforest_idx, pq_idx, hnsw_idx];
    let accuracy = Arc::new(Mutex::new(Vec::new()));
    let cost = Arc::new(Mutex::new(Vec::new()));
    let base_cost = Arc::new(Mutex::new(Duration::default()));
    for i in 0..indices.len() {
        make_idx_baseline(ns.clone(), &mut indices[i]);
        accuracy.lock().unwrap().push(0.);
        cost.lock().unwrap().push(Duration::default());
    }
    make_idx_baseline(ns.clone(), &mut bf_idx);
    // make_idx_baseline(ns.clone(), &mut ssg_idx);
    // ssg_idx.connectivity_profile();
    // let _guard = pprof::ProfilerGuard::new(100).unwrap();
    for _i in Prgrs::new(0..test_time, 1000).set_length_move(Length::Proportional(0.5)) {
        // (0..test_time).into_par_iter().for_each(|_| {
        let mut rng = rand::thread_rng();

        let target: usize = rng.gen_range(0, ns.len());
        let w = ns.get(target).unwrap();

        let base_start = SystemTime::now();
        let base_result = bf_idx.search_full(&w, 100);
        let mut base_set = HashSet::new();
        for (n, _dist) in base_result.iter() {
            base_set.insert(n.idx().unwrap());
            // println!("{:?}", dist);
        }
        let base_since_the_epoch = SystemTime::now()
            .duration_since(base_start)
            .expect("Time went backwards");
        *base_cost.lock().unwrap() += base_since_the_epoch;

        for j in 0..indices.len() {
            let start = SystemTime::now();
            let result = indices[j].search_full(&w, 100);
            for (n, _dist) in result.iter() {
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

    // if let Ok(report) = guard.report().build() {
    //     let file = File::create("flamegraph.svg").unwrap();
    //     let mut options = pprof::flamegraph::Options::default();
    //     options.image_width = Some(2500);
    //     report.flamegraph_with_options(file, &mut options).unwrap();
    // };
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
            "index: {:?}, avg accuracy: {:?}%, hit {:?}, avg cost {:?} millisecond, nodes size {:?}",
            indices[i].name(),
            a / (test_time as f64),
            a,
            cost.lock().unwrap()[i].as_millis() as f64 / (test_time as f64), indices[i].nodes_size()
        );
    }

    bf_idx.dump("bf_idx.idx", &arguments::Args::new());
    let _bf_idx_v2 =
        bf::bf::BruteForceIndex::<f64, usize>::load("bf_idx.idx", &arguments::Args::new());
    // make_idx_baseline(ns.clone(), &mut ssg_idx);
    // ssg_idx.dump("ssg_idx.idx", &arguments::Args::new());
    // let ssg_idx_v2 = mrng::ssg::SatelliteSystemGraphIndex::<f64, usize>::load(
    //     "ssg_idx.idx",
    //     &arguments::Args::new(),
    // );
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
    let file = File::open("/Users/chenyangyang/rust/fastann/src/bench/glove.6B.50d.txt").unwrap();
    let reader = BufReader::new(file);

    let mut idx = 0;
    for line in reader.lines() {
        if let Ok(l) = line {
            if idx == LINE_SIZE {
                break;
            }
            let split_line = l.split(' ').collect::<Vec<&str>>();
            let word = split_line[0];
            let mut vecs = Vec::with_capacity(split_line.len() - 1);
            for i in 1..split_line.len() {
                vecs.push(split_line[i].parse::<f32>().unwrap());
            }
            words.insert(word.to_string(), idx);
            word_idxs.insert(idx, word.to_string());
            words_vec.push(word.to_string());
            words_train_data.insert(word.to_string(), vecs.clone());
            idx += 1;
            train_data.push(vecs.clone());
            if idx % 100000 == 0 {
                println!("load {:?}", idx);
            }
        }
    }

    let mut bf_idx = Box::new(bf::bf::BruteForceIndex::<f32, usize>::new());
    let _bpforest_idx = Box::new(
        bpforest::bpforest::BinaryProjectionForestIndex::<f32, usize>::new(DIMENSION, 6, -1),
    );
    let mut hnsw_idx = Box::new(hnsw::hnsw::HNSWIndex::<f32, usize>::new(
        DIMENSION, 
        &hnsw::hnsw::HNSWParams::default()
    ));

    let _ivfpq_idx = Box::new(pq::pq::IVFPQIndex::<f32, usize>::new(
        DIMENSION,
        &pq::pq::IVFPQParams::<f32>::default()
        .n_sub(DIMENSION/2)
        .sub_bits(4)
        .train_epoch(100)
    ));

    let mut ivfpq_idx = Box::new(
        pq::pq::IVFPQIndex::<f32, usize>::new(
            DIMENSION,
            &pq::pq::IVFPQParams::<f32>::default()
            .n_sub(DIMENSION/2)
            .sub_bits(4)
            .n_kmeans_center(256)
            .search_n_center(4)
            .train_epoch(100)
        )
    );

    let mut ssg_idx = Box::new(mrng::ssg::SatelliteSystemGraphIndex::<f32, usize>::new(
        DIMENSION,
        &mrng::ssg::SatelliteSystemGraphParams::default(),
    ));

    make_idx_baseline(train_data.clone(), &mut bf_idx);
    // make_idx_baseline(train_data.clone(), &mut bpforest_idx);
    // make_idx_baseline(train_data.clone(), &mut hnsw_idx);
    // make_idx_baseline(train_data.clone(), &mut pq_idx);
    make_idx_baseline(train_data.clone(), &mut ivfpq_idx);
    // make_idx_baseline(train_data, &mut ssg_idx);

    let argument = arguments::Args::new();
    bf_idx.dump("bf_idx.idx", &argument);
    // bpforest_idx.dump("bpforest_idx.idx", &argument);
    // hnsw_idx.dump("hnsw_idx.idx", &argument);
    // pq_idx.dump("pq_idx.idx", &argument);
    ivfpq_idx.dump("ivfpq_idx.idx", &argument);    
    // ssg_idx.dump("ssg_idx.idx", &argument);
}

pub fn run() {
    let mut words = HashMap::new();
    let mut word_idxs = HashMap::new();
    let mut words_vec = Vec::new();
    let mut train_data = Vec::new();
    let mut words_train_data = HashMap::new();
    let file = File::open("/Users/chenyangyang/rust/fastann/src/bench/glove.6B.50d.txt").unwrap();
    let reader = BufReader::new(file);

    let mut idx = 0;
    for line in reader.lines() {
        if let Ok(l) = line {
            if idx == LINE_SIZE {
                println!("done {:?}", idx);
                break;
            }
            let split_line = l.split(' ').collect::<Vec<&str>>();
            let word = split_line[0];
            let mut vecs = Vec::with_capacity(split_line.len() - 1);
            for i in 1..split_line.len() {
                vecs.push(split_line[i].parse::<f32>().unwrap());
            }
            words.insert(word.to_string(), idx);
            word_idxs.insert(idx, word.to_string());
            words_vec.push(word.to_string());
            words_train_data.insert(word.to_string(), vecs.clone());
            idx += 1;
            train_data.push(vecs.clone());
            if idx % 1000 == 0 {
                println!("load {:?}", idx);
            }
        }
    }
    let argument = arguments::Args::new();
    let bf_idx =
        Box::new(bf::bf::BruteForceIndex::<f32, usize>::load("bf_idx.idx", &argument).unwrap());
    // let _bpforest_idx = Box::new(
    //     bpforest::bpforest::BinaryProjectionForestIndex::<f32, usize>::load(
    //         "bpforest_idx.idx",
    //         &argument,
    //     )
    //     .unwrap(),
    // );
    // let _hnsw_idx =
    //     Box::new(hnsw::hnsw::HNSWIndex::<f32, usize>::load("hnsw_idx.idx", &argument).unwrap());

    // let _pq_idx = Box::new(pq::pq::PQIndex::<f32, usize>::load("pq_idx.idx", &argument).unwrap());
    let _pq_idx = Box::new(pq::pq::PQIndex::<f32, usize>::load("pq_idx.idx", &argument).unwrap());
    let _ivfpq_idx = Box::new(
        pq::pq::IVFPQIndex::<f32, usize>::load("ivfpq_idx.idx", &argument).unwrap()
    );
    // let _ssg_idx = Box::new(
    //     mrng::ssg::SatelliteSystemGraphIndex::<f32, usize>::load("ssg_idx.idx", &argument).unwrap(),
    // );

    let indices: Vec<Box<dyn ANNIndex<f32, usize>>> =
        // vec![bpforest_idx, pq_idx, ssg_idx, hnsw_idx];
        // vec![_hnsw_idx];
        // vec![_pq_idx];
        vec![_ivfpq_idx];

    const K: i32 = 1000;
    let words: Vec<usize> = (0..K)
        .map(|_i| {
            let mut rng = rand::thread_rng();
            let target_word: usize = rng.gen_range(1, words_vec.len());
            let w = words.get(&words_vec[target_word]).unwrap();
            *w
        })
        .collect();

    let results: Vec<HashSet<usize>> = words
        .iter()
        .map(|w| {
            bf_idx
                .search_full(&train_data[*w as usize], 100)
                .into_iter()
                .map(|x| x.0.idx().unwrap())
                .collect()
        })
        .collect();

    for idx in indices.iter() {
        let start = SystemTime::now();
        // println!("hioyipppppp");
        // let guard = pprof::ProfilerGuard::new(50).unwrap();
        // println!("hioyioiohio");
        let mut accuracy = 0;
        words.iter().zip(0..words.len()).for_each(|(w, i)| {
            // println!("hioyo {:?} {:?}", i, words.len());
            let result = idx.search(&train_data[*w as usize], 10);
            // println!("hio {:?} {:?}", i, words.len());
            for n in result.iter() {
                if results[i].contains(n) {
                    accuracy += 1;
                }
            }
        });

        // if let Ok(report) = guard.report().build() {
        //     let file = File::create(format!("flamegraph2.{}.svg", idx.name())).unwrap();
        //     report.flamegraph(file).unwrap();
        // };
        let since_the_epoch = SystemTime::now()
            .duration_since(start)
            .expect("Time went backwards");
        println!(
            "{:?}: {:?} accuracy: {}",
            idx.name(),
            since_the_epoch,
            accuracy
        );
    }
}

fn make_idx_baseline<E: core::node::FloatElement, T: ANNIndex<E, usize> + ?Sized>(
    embs: Vec<Vec<E>>,
    idx: &mut Box<T>,
) {
    let start = SystemTime::now();
    for i in 0..embs.len() {
        idx.add_node(&core::node::Node::<E, usize>::new_with_idx(&embs[i], i));
    }
    idx.build(core::metrics::Metric::Euclidean).unwrap();
    let since_start = SystemTime::now()
        .duration_since(start)
        .expect("Time went backwards");

    println!(
        "index {:?} build time {:?} ms",
        idx.name(),
        since_start.as_millis() as f64
    );
}
