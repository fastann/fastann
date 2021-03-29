use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use fixedbitset::FixedBitSet;
use rand::prelude::*;

#[cfg(feature = "without_std")]
use hashbrown::HashSet;

use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "without_std"))]
use std::collections::HashSet;
use std::collections::VecDeque;

use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::sync::{Arc, Mutex};

#[derive(Debug, Serialize, Deserialize)]
pub struct SatelliteSystemGraphIndex<E: node::FloatElement, T: node::IdxType> {
    #[serde(skip_serializing, skip_deserializing)]
    nodes: Vec<Box<node::Node<E, T>>>,
    tmp_nodes: Vec<node::Node<E, T>>, // only use for serialization scene
    mt: metrics::Metric,
    dimension: usize,
    neighbor_size: usize,
    index_size: usize,
    graph: Vec<Vec<usize>>,
    knn_graph: Vec<Vec<usize>>,
    init_k: usize, // as knn's k
    root_nodes: Vec<usize>,
    width: usize,
    angle: E,
    threshold: E,
    root_size: usize,
}

impl<E: node::FloatElement, T: node::IdxType> SatelliteSystemGraphIndex<E, T> {
    pub fn new(
        dimension: usize,
        neighbor_size: usize,
        init_k: usize,
        index_size: usize,
        angle: E,
        root_size: usize,
    ) -> SatelliteSystemGraphIndex<E, T> {
        SatelliteSystemGraphIndex::<E, T> {
            nodes: Vec::new(),
            tmp_nodes: Vec::new(),
            mt: metrics::Metric::Unknown,
            dimension,
            neighbor_size,
            init_k,
            graph: Vec::new(),
            knn_graph: Vec::new(),
            root_nodes: Vec::new(),
            width: 0,
            index_size,
            angle,
            threshold: (angle / E::from_f32(180.0).unwrap() * E::PI()).cos(),
            root_size,
        }
    }

    fn build_knn_graph(&mut self) {
        let tmp_graph = Arc::new(Mutex::new(vec![vec![0]; self.nodes.len()]));
        (0..self.nodes.len()).into_par_iter().for_each(|n| {
            let item = &self.nodes[n];
            let mut heap = BinaryHeap::with_capacity(self.init_k);
            for i in 0..self.nodes.len() {
                if i == n {
                    continue;
                }
                heap.push(neighbor::Neighbor::new(
                    i,
                    item.metric(&self.nodes[i], self.mt).unwrap(),
                ));
                if heap.len() > self.init_k {
                    heap.pop();
                }
            }
            let mut tmp = Vec::with_capacity(heap.len());
            while !heap.is_empty() {
                tmp.push(heap.pop().unwrap().idx());
            }

            tmp_graph.lock().unwrap()[n] = tmp;
        });
        self.graph = tmp_graph.lock().unwrap().to_vec();
        self.knn_graph = tmp_graph.lock().unwrap().to_vec();
    }

    fn get_random_nodes_idx_lite(&self, indices: &mut [usize]) {
        let mut rng = rand::thread_rng();
        (0..indices.len()).for_each(|i| {
            indices[i] = rng.gen_range(0, self.nodes.len() - indices.len());
        });
    }

    fn get_point_neighbor_size_neighbors(
        &self,
        q: usize,
        pool: &mut Vec<neighbor::Neighbor<E, usize>>,
    ) {
        let mut flags = HashSet::with_capacity(self.neighbor_size);

        flags.insert(q);
        for i in 0..self.graph[q].len() {
            let nid = self.graph[q][i];
            for nn in 0..self.graph[nid].len() {
                if nn == i {
                    continue;
                }
                let nn_id = self.graph[nid][nn];
                if flags.contains(&nn_id) {
                    continue;
                }
                flags.insert(nn_id);
                let dist = self.nodes[q].metric(&self.nodes[nn_id], self.mt).unwrap();
                pool.push(neighbor::Neighbor::new(nn_id, dist));
                if pool.len() >= self.neighbor_size {
                    return;
                }
            }
        }
    }

    fn expand_connectivity(&mut self) {
        let range = self.index_size;

        let mut ids: Vec<usize> = (0..self.nodes.len()).collect();
        ids.shuffle(&mut thread_rng());
        for id in ids.iter().take(self.root_size) {
            self.root_nodes.push(*id);
        }

        (0..self.root_size).for_each(|i| {
            let root_id = self.root_nodes[i];
            let mut flags = HashSet::new();
            let mut my_queue = VecDeque::new();
            my_queue.push_back(root_id);
            flags.insert(root_id);

            let mut unknown_set: Vec<usize> = Vec::with_capacity(1);
            while !unknown_set.is_empty() {
                while !my_queue.is_empty() {
                    let q_front = my_queue.pop_front().unwrap();

                    for j in 0..self.graph[q_front].len() {
                        let child = self.graph[q_front][j];
                        if flags.contains(&child) {
                            continue;
                        }
                        flags.insert(child);
                        my_queue.push_back(child);
                    }
                }
                unknown_set.clear();
                for j in 0..self.nodes.len() {
                    if flags.contains(&j) {
                        continue;
                    }
                    unknown_set.push(j);
                }
                if !unknown_set.is_empty() {
                    for j in 0..self.nodes.len() {
                        if flags.contains(&j) && self.graph[j].len() < range {
                            self.graph[j].push(unknown_set[0]);
                            break;
                        }
                    }
                    my_queue.push_back(unknown_set[0]);
                    flags.insert(unknown_set[0]);
                }
            }
        });
    }

    fn link_each_nodes(&mut self, cut_graph: &mut Vec<neighbor::Neighbor<E, usize>>) {
        let mut pool = Vec::new();
        (0..self.nodes.len()).for_each(|i| {
            pool.clear();
            self.get_point_neighbor_size_neighbors(i, &mut pool); // get related one
            self.prune_graph(i, &mut pool, self.threshold, cut_graph);
        });
        (0..self.nodes.len()).for_each(|i| {
            self.inter_insert(i, self.index_size, cut_graph);
        });
    }

    fn prune_graph(
        &mut self,
        query_id: usize,
        related_ids_pool: &mut Vec<neighbor::Neighbor<E, usize>>,
        threshold: E,
        cut_graph: &mut Vec<neighbor::Neighbor<E, usize>>,
    ) {
        self.width = self.index_size;
        let mut start = 0;
        let mut flags = HashSet::with_capacity(related_ids_pool.len());
        for iter in related_ids_pool.iter() {
            flags.insert(iter.idx());
        }
        self.graph[query_id].iter().for_each(|linked_id| {
            if flags.contains(linked_id) {
                return;
            }
            related_ids_pool.push(neighbor::Neighbor::new(
                *linked_id,
                self.nodes[query_id]
                    .metric(&self.nodes[*linked_id], self.mt)
                    .unwrap(),
            ));
        });

        related_ids_pool.sort_unstable();
        let mut result = Vec::new();
        if related_ids_pool[start].idx() == query_id {
            start += 1;
        }
        result.push(related_ids_pool[start].clone());

        start += 1;
        while result.len() < self.index_size && start < related_ids_pool.len() {
            let p = &related_ids_pool[start];
            let mut occlude = false;
            // TODO: check every metrics, and decide use euclidean forcibly.
            for iter in result.iter() {
                if p.idx() == iter.idx() {
                    // stop early
                    occlude = true;
                    break;
                }
                let djk = self.nodes[iter.idx()]
                    .metric(&self.nodes[p.idx()], self.mt)
                    .unwrap();
                let cos_ij = (p.distance().powi(2) + iter.distance().powi(2) - djk.powi(2))
                    / (E::from_usize(2).unwrap() * (p.distance() * iter.distance()));

                if cos_ij > threshold {
                    occlude = true;
                    break;
                }
            }
            if !occlude {
                result.push(p.clone());
            }
            start += 1;
        }

        for t in 0..result.len() {
            cut_graph[t + query_id * self.index_size]._idx = result[t].idx();
            cut_graph[t + query_id * self.index_size]._distance = result[t].distance();
        }
        if result.len() < self.index_size {
            for i in result.len()..self.index_size {
                cut_graph[query_id * self.index_size + i]._distance = E::max_value();
                cut_graph[query_id * self.index_size + i]._idx = self.nodes.len();
                // means not exist
            }
        }
    }

    // to handle neighbor's graph
    fn inter_insert(
        &self,
        n: usize,
        range: usize,
        cut_graph: &mut Vec<neighbor::Neighbor<E, usize>>,
    ) {
        (0..range).for_each(|i| {
            if cut_graph[i + n].distance() == E::max_value() {
                return;
            }

            let sn = neighbor::Neighbor::new(n, cut_graph[i + n].distance()); // distance of n to i
            let des = cut_graph[i + n].idx();
            let mut temp_pool = Vec::new();
            let mut dup = false;

            for j in 0..range {
                if cut_graph[j + des * self.index_size].distance() == E::max_value() {
                    break;
                }
                if n == cut_graph[j + des * self.index_size].idx() {
                    // neighbor and point meet
                    dup = true;
                    break;
                }
                temp_pool.push(cut_graph[j + des * self.index_size].clone()); // neighbor's neighbor
            }

            if dup {
                return;
            }

            temp_pool.push(sn.clone());
            if temp_pool.len() > range {
                let mut result = Vec::new();
                let mut start = 0;
                temp_pool.sort_unstable();
                result.push(temp_pool[start].clone());
                start += 1;
                while result.len() < range && start < temp_pool.len() {
                    let p = &temp_pool[start];
                    let mut occlude = false;
                    for t in 0..result.len() {
                        if p.idx() == result[t].idx() {
                            occlude = true;
                            break;
                        }
                        let djk = self.nodes[result[t].idx()]
                            .metric(&self.nodes[p.idx()], self.mt)
                            .unwrap();
                        let cos_ij = (p.distance().powi(2) + result[t].distance().powi(2)
                            - djk.powi(2))
                            / (E::from_usize(2).unwrap() * (p.distance() * result[t].distance()));

                        if cos_ij > self.threshold {
                            occlude = true;
                            break;
                        }
                    }
                    if !occlude {
                        result.push(p.clone());
                    }
                    start += 1;
                }
                for t in 0..result.len() {
                    cut_graph[t + des * self.index_size] = result[t].clone();
                }

                if result.len() < range {
                    cut_graph[result.len() + des * self.index_size]._distance = E::max_value();
                }
            } else {
                for t in 0..range {
                    if cut_graph[t + des * self.index_size].distance() == E::max_value() {
                        cut_graph[t + des * self.index_size] = sn.clone();
                        if (t + 1) < range {
                            cut_graph[t + des * self.index_size]._distance = E::max_value();
                            break;
                        }
                    }
                }
            }
        });
    }

    fn build(&mut self) {
        self.build_knn_graph();

        let mut cut_graph: Vec<neighbor::Neighbor<E, usize>> =
            Vec::with_capacity(self.nodes.len() * self.index_size);
        for i in 0..self.nodes.len() * self.index_size {
            cut_graph.push(neighbor::Neighbor::<E, usize>::new(i, E::float_zero()));
        }
        self.link_each_nodes(&mut cut_graph);

        for i in 0..self.nodes.len() {
            let mut pool_size = 0;
            for j in 0..self.index_size {
                if cut_graph[i * self.index_size + j].distance() == E::max_value() {
                    break;
                }
                pool_size = j;
            }
            pool_size += 1;
            self.graph[i] = Vec::with_capacity(pool_size);
            for j in 0..pool_size {
                self.graph[i].push(cut_graph[i * self.index_size + j].idx());
            }
        }

        self.expand_connectivity();

        let mut max = 0;
        let mut min = self.nodes.len();
        let mut avg: f32 = 0.;
        for t in 0..self.nodes.len() {
            let size = self.graph[t].len();
            max = if max < size { size } else { max };
            min = if min > size { size } else { min };
            avg += size as f32;
        }
        avg /= 1.0 * self.nodes.len() as f32;
        println!(
            "stat: k: {:?}, max {:?}, min {:?}, avg {:?}",
            self.init_k, max, min, avg
        );
    }

    fn search(
        &self,
        query: &node::Node<E, T>,
        k: usize,
        _args: &arguments::Args,
    ) -> Vec<(node::Node<E, T>, E)> {
        let mut l = k;
        if l < self.root_nodes.len() {
            l = self.root_nodes.len();
        }
        let mut init_ids = vec![0; l];
        // let mut search_flags = HashSet::with_capacity(self.nodes.len());
        let mut search_flags = FixedBitSet::with_capacity(self.nodes.len());
        let mut heap: BinaryHeap<neighbor::Neighbor<E, usize>> = BinaryHeap::new(); // max-heap
        let mut search_queue = VecDeque::new();

        (0..self.root_nodes.len()).for_each(|i| {
            init_ids[i] = self.root_nodes[i];
        });
        self.get_random_nodes_idx_lite(&mut init_ids[self.root_nodes.len()..]);

        (0..l).for_each(|i| {
            let id = init_ids[i];
            let dist = self.nodes[id].metric(query, self.mt).unwrap();
            heap.push(neighbor::Neighbor::new(id, dist));
            search_queue.extend(&self.graph[id]);
            search_flags.insert(id);
        });

        // greedy BFS search
        while !search_queue.is_empty() {
            let id = search_queue.pop_front().unwrap();
            if search_flags.contains(id) {
                continue;
            }

            let dist = self.nodes[id].metric(query, self.mt).unwrap();
            if dist < heap.peek().unwrap().distance() {
                heap.pop();
                heap.push(neighbor::Neighbor::new(id, dist));
                search_queue.extend(&self.graph[id]);
            }
            search_flags.insert(id);
        }

        let mut result = Vec::new();
        while !heap.is_empty() {
            let tmp = heap.pop().unwrap();
            result.push((*self.nodes[tmp.idx()].clone(), tmp.distance()));
        }
        result.reverse();
        result
    }

    fn check_edge(&self, h: usize, t: usize) -> bool {
        let mut flag = true;
        for i in 0..self.graph[h].len() {
            if t == self.graph[h][i] {
                flag = false;
            }
        }
        flag
    }

    pub fn connectivity_profile(&self) {
        let mut visited = HashSet::with_capacity(self.nodes.len());
        let mut queue = VecDeque::new();

        queue.push_back(0);
        while !queue.is_empty() {
            let id = queue.pop_front().unwrap();
            if visited.contains(&id) {
                continue;
            }

            for x in 0..self.graph[id].len() {
                queue.push_back(self.graph[id][x]);
                if self.graph[id][x] > self.nodes.len() {
                    // println!("{:?} {:?} {:?}", self.graph[id][x], self.graph[id], id);
                }
            }
            visited.insert(id);
        }

        println!("connectivity: {:?} {:?}", self.nodes.len(), visited.len());
    }
}

impl<E: node::FloatElement + DeserializeOwned, T: node::IdxType + DeserializeOwned>
    ann_index::SerializableIndex<E, T> for SatelliteSystemGraphIndex<E, T>
{
    fn load(path: &str, _args: &arguments::Args) -> Result<Self, &'static str> {
        let file = File::open(path).unwrap_or_else(|_| panic!("unable to open file {:?}", path));
        let mut instance: SatelliteSystemGraphIndex<E, T> =
            bincode::deserialize_from(&file).unwrap();
        instance.nodes = instance
            .tmp_nodes
            .iter()
            .map(|x| Box::new(x.clone()))
            .collect();
        Ok(instance)
    }

    fn dump(&mut self, path: &str, _args: &arguments::Args) -> Result<(), &'static str> {
        self.tmp_nodes = self.nodes.iter().map(|x| *x.clone()).collect();
        let encoded_bytes = bincode::serialize(&self).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&encoded_bytes)
            .unwrap_or_else(|_| panic!("unable to write file {:?}", path));
        Result::Ok(())
    }
}

impl<E: node::FloatElement, T: node::IdxType> ann_index::ANNIndex<E, T>
    for SatelliteSystemGraphIndex<E, T>
{
    fn construct(&mut self, mt: metrics::Metric) -> Result<(), &'static str> {
        self.mt = mt;
        self.build();

        Result::Ok(())
    }
    fn add_node(&mut self, item: &node::Node<E, T>) -> Result<(), &'static str> {
        self.nodes.push(Box::new(item.clone()));
        Result::Ok(())
    }
    fn once_constructed(&self) -> bool {
        true
    }
    fn reconstruct(&mut self, _mt: metrics::Metric) {}
    fn node_search_k(
        &self,
        item: &node::Node<E, T>,
        k: usize,
        args: &arguments::Args,
    ) -> Vec<(node::Node<E, T>, E)> {
        self.search(&item, k, &args)
    }

    fn name(&self) -> &'static str {
        "SatelliteSystemGraphIndex"
    }

    fn nodes_size(&self) -> usize {
        self.nodes.len()
    }
}
