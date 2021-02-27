use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use core::cmp::Reverse;
use rand::prelude::*;
extern crate num;
use core::cmp::Ordering;
use std::collections::VecDeque;

// TODO: migrate to neighbor
#[derive(Default, Clone, PartialEq, Debug)]
pub struct SubNeighbor<E: node::FloatElement, T: node::IdxType> {
    pub _idx: T,
    pub _distance: E,
    pub flag: bool,
}

impl<E: node::FloatElement, T: node::IdxType> SubNeighbor<E, T> {
    pub fn new(idx: T, distance: E, flag: bool) -> SubNeighbor<E, T> {
        return SubNeighbor {
            _idx: idx,
            _distance: distance,
            flag: flag,
        };
    }

    pub fn idx(&self) -> T {
        self._idx.clone()
    }

    pub fn distance(&self) -> E {
        self._distance
    }
}

impl<E: node::FloatElement, T: node::IdxType> Ord for SubNeighbor<E, T> {
    fn cmp(&self, other: &SubNeighbor<E, T>) -> Ordering {
        self._distance.partial_cmp(&other._distance).unwrap()
    }
}

impl<E: node::FloatElement, T: node::IdxType> PartialOrd for SubNeighbor<E, T> {
    fn partial_cmp(&self, other: &SubNeighbor<E, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E: node::FloatElement, T: node::IdxType> Eq for SubNeighbor<E, T> {}

pub struct SatelliteSystemGraphIndex<E: node::FloatElement, T: node::IdxType> {
    nodes: Vec<Box<node::Node<E, T>>>,
    mt: metrics::Metric,
    dimension: usize,
    L: usize,
    graph: Vec<Vec<usize>>, // as final_graph_
    init_k: usize,          // as knn's k
    ep_: usize,
    eps: Vec<usize>,
}

impl<E: node::FloatElement, T: node::IdxType> SatelliteSystemGraphIndex<E, T> {
    pub fn new(dimension: usize, L: usize, init_k: usize) -> SatelliteSystemGraphIndex<E, T> {
        SatelliteSystemGraphIndex::<E, T> {
            nodes: Vec::new(),
            mt: metrics::Metric::Unknown,
            dimension: dimension,
            L: L,
            init_k: init_k,
            graph: Vec::new(),
            ep_: 0,
            eps: Vec::new(),
        }
    }

    fn initialize_graph(&mut self) {
        let mut center = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            center.push(E::float_zero());
        }
        for i in 0..self.nodes.len() {
            for j in 0..self.dimension {
                center[j] += self.nodes[i].vectors()[j];
            }
        }
        for j in 0..self.dimension {
            center[j] /= E::from_usize(self.nodes.len()).unwrap();
        }

        let mut tmp: Vec<SubNeighbor<E, usize>> = Vec::new();
        let mut pool: Vec<SubNeighbor<E, usize>> = Vec::new();
        let n = node::Node::new(&center);
        self.get_point_neighbors(&n, &mut tmp, &mut pool);
        self.ep_ = tmp[0].idx();
    }

    fn build_knn_graph(&mut self) {
        self.graph = Vec::new();
        for n in 0..self.nodes.len() {
            let item = &self.nodes[n];
            let mut heap = BinaryHeap::new();
            for i in 0..self.nodes.len() {
                heap.push(neighbor::Neighbor::new(
                    i,
                    item.metric(&self.nodes[i], self.mt).unwrap(),
                ));
                if heap.len() > self.init_k {
                    heap.pop();
                }
            }
            self.graph.push(Vec::new());
            while !heap.is_empty() {
                self.graph[n].push(heap.pop().unwrap().idx());
            }
        }
    }

    fn get_random_nodes_idx(&self, indices: &mut [usize]) {
        let mut rng = rand::thread_rng();
        for i in 0..indices.len() {
            indices[i] = rng.gen_range(0, self.nodes.len() - indices.len());
        }
        indices.sort();
        for i in 1..indices.len() {
            if indices[i] <= indices[i - 1] {
                indices[i] = indices[i - 1] + 1;
            }
        }
        let offset: usize = rng.gen_range(0, self.nodes.len());
        for i in 0..indices.len() {
            indices[i] = (indices[i] + offset) % self.nodes.len();
        }
    }

    fn insert_into_pools(
        &self,
        addr: &mut Vec<SubNeighbor<E, usize>>,
        k: usize,
        nn: &SubNeighbor<E, usize>,
    ) -> usize {
        let mut left = 0;
        let mut right = k - 1;
        if addr[left].distance() > nn.distance() {
            addr[left] = nn.clone();
            return left;
        }
        if addr[right].distance() < nn.distance() {
            addr[k] = nn.clone();
            return k;
        }
        while left < right - 1 {
            let mid = (left + right) / 2;
            if addr[mid].distance() > nn.distance() {
                right = mid;
            } else {
                left = mid;
            }
        }
        while left > 0 {
            if addr[left].distance() < nn.distance() {
                break;
            }
            if addr[left].idx() == nn.idx() {
                return k + 1;
            }
            left -= 1;
        }
        if addr[left].idx() == nn.idx() || addr[right].idx() == nn.idx() {
            return k + 1;
        }
        addr[right] = nn.clone();
        return right;
    }

    fn get_point_neighbors(
        &self,
        item: &node::Node<E, T>,
        return_set: &mut Vec<SubNeighbor<E, usize>>,
        full_set: &mut Vec<SubNeighbor<E, usize>>,
    ) {
        let mut init_ids = vec![0; self.L];
        self.get_random_nodes_idx(&mut init_ids);
        let mut flags = vec![false; self.nodes.len()]; // as bitmap;
        let mut _L = 0;
        for id in init_ids.iter() {
            if *id > self.nodes.len() {
                continue;
            }
            return_set.push(SubNeighbor::new(
                id.clone(),
                item.metric(&self.nodes[*id], self.mt).unwrap(),
                true,
            ));
            flags[*id] = true;
            _L += 1;
        }
        return_set.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut k = 0;
        while k < _L {
            let mut nk = _L;
            if return_set[k].flag {
                return_set[k].flag = false;
                let n = return_set[k].idx();

                for m in 0..self.graph[n].len() {
                    let id = self.graph[n][m];
                    if flags[id] {
                        continue;
                    }
                    flags[id] = true;
                    let dist = item.metric(&self.nodes[id], self.mt).unwrap();
                    full_set.push(SubNeighbor::new(id, dist, true));
                    if dist > return_set[self.L - 1].distance() {
                        continue;
                    }
                    let r = self.insert_into_pools(return_set, _L, &full_set[full_set.len() - 1]);
                    if _L + 1 < return_set.len() {
                        _L += 1;
                    }
                    if r < nk {
                        nk = r;
                    }
                }
                if nk <= k {
                    k = nk;
                } else {
                    k += 1;
                }
            }
        }
    }

    fn get_point_neighbors_v2(&self, q:usize, pool: &mut Vec<SubNeighbor<E, usize>>) {
        let flags = vec![false; self.nodes.len()];
        let L = 5;

        flags[q] = true;
        for i in 0..self.graph[q].len() {
            let nid = self.graph[q][i];
            for nn in self.graph[nid].len() {
                let nnid = self.graph[nid][nn];
                if flags[nnid] {
                    continue;
                }
                flags[nnid] = true;
                let dist = self.nodes[q].metric(&self.nodes[nnid], self.mt).unwrap();
                pool.push(SubNeighbor::new(nnid,dist,true));
                if pool.len() >= L {
                    return;
                }
            }
        }
    }

    fn expand_dfs(&mut self) {
        let n_try = 5; // TODO: Params
        let range = 10; // TODO: Params

        let mut ids: Vec<usize> = (0..self.nodes.len()).collect();
        ids.shuffle(&mut thread_rng());
        for i in 0..n_try {
            self.eps.push(ids[i]);
        }

        for i in 0..n_try {
            let root_id = self.eps[i];
            let mut flags = vec![false; self.nodes.len()];
            let mut my_queue = VecDeque::new();
            my_queue.push_back(root_id);
            flags[root_id] = true;

            let mut unknown_set: Vec<usize> = Vec::with_capacity(1);
            while unknown_set.len() > 0 {
                while (!my_queue.is_empty()) {
                    let q_front = my_queue.pop_front().unwrap();

                    for j in 0..self.graph[q_front].len() {
                        let child = self.graph[q_front][j];
                        if flags[child] {
                            continue;
                        }
                        flags[child] = true;
                        my_queue.push_back(child);
                    }
                }
                unknown_set.clear();
                for j in 0..self.nodes.len() {
                    if flags[j] {
                        continue;
                    }
                    unknown_set.push(j);
                }
                if unknown_set.len() > 0 {
                    for j in 0..self.nodes.len() {
                        if flags[j] && self.graph[j].len() < range {
                            self.graph[j].push(unknown_set[0]);
                            break;
                        }
                    }
                    my_queue.push_back(unknown_set[0]);
                    flags[unknown_set[0]] = true;
                }
            }
        }
    }

    fn link(&self, cut_graph: &Vec<node::Neighbor<E, usize>>) {
        let range = 5;
        let angle :f32= 0.5;
        let threshold : f32= (angle/180.0 * 3.14).cos();
        let pool = Vec::new();
        let tmp = Vec::new();
        for i in 0..self.nodes.len() {
            pool.clear();
            tmp.clear();

        }
    }

    fn build(&mut self) {
        let range = 5;

        self.build_knn_graph();
        self.initialize_graph();

        let cut_graph :Vec<node::Neighbor<E, usize>> = Vec::with_capacity(self.nodes.len() * range);

    }
}

impl<E: node::FloatElement, T: node::IdxType> ann_index::ANNIndex<E, T>
    for SatelliteSystemGraphIndex<E, T>
{
    fn construct(&mut self, mt: metrics::Metric) -> Result<(), &'static str> {
        self.mt = mt;
        Result::Ok(())
    }
    fn add_node(&mut self, item: &node::Node<E, T>) -> Result<(), &'static str> {
        self.nodes.push(Box::new(item.clone()));
        Result::Ok(())
    }
    fn once_constructed(&self) -> bool {
        true
    }
    fn reconstruct(&mut self, mt: metrics::Metric) {}
    fn node_search_k(
        &self,
        item: &node::Node<E, T>,
        k: usize,
        args: &arguments::Args,
    ) -> Vec<(node::Node<E, T>, E)> {
        return Vec::new();
    }

    fn load(&self, path: &str) -> Result<(), &'static str> {
        Result::Ok(())
    }

    fn dump(&self, path: &str) -> Result<(), &'static str> {
        Result::Ok(())
    }

    fn name(&self) -> &'static str {
        "BruteForceIndex"
    }
}
