use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use rand::prelude::*;
extern crate num;
use bincode;
use core::cmp::Ordering;
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// TODO: migrate to neighbor
#[derive(Default, Clone, Copy, PartialEq, Debug)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct SatelliteSystemGraphIndex<E: node::FloatElement, T: node::IdxType> {
    #[serde(skip_serializing, skip_deserializing)]
    nodes: Vec<Box<node::Node<E, T>>>,
    tmp_nodes: Vec<node::Node<E, T>>, // only use for serialization scene
    mt: metrics::Metric,
    dimension: usize,
    L: usize,
    index_size: usize,      // as R
    graph: Vec<Vec<usize>>, // as final_graph_
    knn_graph: Vec<Vec<usize>>,
    init_k: usize,          // as knn's k
    root_nodes: Vec<usize>, // eps
    width: usize,
    opt_graph: Vec<Vec<usize>>,
    angle: E,
    threshold: E,
    n_try: usize,
}

impl<E: node::FloatElement, T: node::IdxType> SatelliteSystemGraphIndex<E, T> {
    pub fn new(
        dimension: usize,
        L: usize,
        init_k: usize,
        index_size: usize,
        angle: E,
        n_try: usize,
    ) -> SatelliteSystemGraphIndex<E, T> {
        SatelliteSystemGraphIndex::<E, T> {
            nodes: Vec::new(),
            tmp_nodes: Vec::new(),
            mt: metrics::Metric::Unknown,
            dimension: dimension,
            L: L,
            init_k: init_k,
            graph: Vec::new(),
            knn_graph: Vec::new(),
            root_nodes: Vec::new(),
            width: 0,
            opt_graph: Vec::new(),
            index_size: index_size,
            angle: angle,
            threshold: (angle / E::from_f32(180.0).unwrap() * E::from_f32(3.14).unwrap()).cos(),
            n_try: n_try,
        }
    }

    fn initialize_graph(&mut self) {
        let mut center = Vec::with_capacity(self.dimension);
        for _i in 0..self.dimension {
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
    }

    fn build_knn_graph(&mut self) {
        let tmp_graph = Arc::new(Mutex::new(vec![vec![0]; self.nodes.len()]));
        (0..self.nodes.len()).into_par_iter().for_each(|n| {
            let item = &self.nodes[n];
            let mut heap = BinaryHeap::with_capacity(self.init_k);
            for i in 0..self.nodes.len() {
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
        let mut return_set_flags = vec![false; self.nodes.len()];
        for i in 0..init_ids.len() {
            let id = init_ids[i];
            if id > self.nodes.len() {
                continue;
            }
            return_set.push(SubNeighbor::new(
                id.clone(),
                item.metric(&self.nodes[id], self.mt).unwrap(),
                true,
            ));
            return_set_flags[id] = true;
            flags[id] = true;
            _L += 1;
        }
        return_set.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut k = 0;
        while k < _L {
            let mut nk = _L;
            if return_set_flags[k] {
                return_set_flags[k] = false;
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
                    let r = self.insert_into_pools(return_set, _L, &full_set[full_set.len() - 1]); // TODO: 考虑用堆
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

    fn get_point_neighbors_v2(&self, q: usize, pool: &mut Vec<SubNeighbor<E, usize>>) {
        let mut flags = vec![false; self.nodes.len()];
        let L = 5;

        flags[q] = true;
        for i in 0..self.graph[q].len() {
            let nid = self.graph[q][i];
            for nn in 0..self.graph[nid].len() {
                let nnid = self.graph[nid][nn];
                if flags[nnid] {
                    continue;
                }
                flags[nnid] = true;
                let dist = self.nodes[q].metric(&self.nodes[nnid], self.mt).unwrap();
                pool.push(SubNeighbor::new(nnid, dist, true));
                if pool.len() >= L {
                    return;
                }
            }
        }
    }

    fn expand_connectivity(&mut self) {
        let n_try = self.n_try;
        let range = self.index_size;

        let mut ids: Vec<usize> = (0..self.nodes.len()).collect();
        ids.shuffle(&mut thread_rng());
        for i in 0..n_try {
            self.root_nodes.push(ids[i]);
        }

        // TODO: parallel
        (0..n_try).for_each(|i| {
            let root_id = self.root_nodes[i];
            let mut flags = HashSet::new();
            let mut my_queue = VecDeque::new();
            my_queue.push_back(root_id);
            flags.insert(root_id);

            let mut unknown_set: Vec<usize> = Vec::with_capacity(1);
            while unknown_set.len() > 0 {
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
                if unknown_set.len() > 0 {
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

    fn link_each_nodes(
        &mut self,
        cut_graph: &mut Vec<neighbor::Neighbor<E, usize>>,
    ) -> Result<(), &'static str> {
        let range = self.index_size;
        let angle = E::from_f32(0.5).unwrap();
        let threshold = self.threshold;
        let mut pool = Vec::new();
        let mut tmp: Vec<SubNeighbor<E, usize>> = Vec::new();
        for i in 0..self.nodes.len() {
            pool.clear();
            tmp.clear();
            self.get_point_neighbors_v2(i, &mut pool); // get related one
            self.prune_graph(i, &mut pool, threshold, cut_graph);
        }
        for i in 0..self.nodes.len() {
            self.inter_insert(i, range, cut_graph);
        }
        Result::Ok(())
    }

    fn prune_graph(
        &mut self,
        query_id: usize,
        related_ids_pool: &mut Vec<SubNeighbor<E, usize>>,
        threshold: E,
        cut_graph: &mut Vec<neighbor::Neighbor<E, usize>>,
    ) {
        let range = 5;
        self.width = range;
        let mut start = 0;
        let mut flags = vec![false; self.nodes.len()]; // TODO: hashset
        for i in 0..related_ids_pool.len() {
            flags[related_ids_pool[i].idx()] = true;
        }
        for linked_idx in 0..self.graph[query_id].len() {
            let linked_id = self.graph[query_id][linked_idx];
            if flags[linked_id] {
                continue;
            }
            related_ids_pool.push(SubNeighbor::new(
                linked_id,
                self.nodes[query_id]
                    .metric(&self.nodes[linked_id], self.mt)
                    .unwrap(),
                true,
            ));
        }
        related_ids_pool.sort();
        let mut result = Vec::new();
        if related_ids_pool[start].idx() == query_id {
            start += 1;
        }
        result.push(related_ids_pool[start]);

        start += 1;
        while result.len() < range && start < related_ids_pool.len() {
            let p = related_ids_pool[start];
            let mut occlude = false;
            for t in 0..result.len() {
                if p.idx() == result[t].idx() {
                    occlude = true;
                    break;
                }
                let djk = self.nodes[result[t].idx()]
                    .metric(&self.nodes[p.idx()], self.mt)
                    .unwrap();
                let cos_ij = (p.distance() + result[t].distance() - djk)
                    / E::from_usize(2).unwrap()
                    / (p.distance() * result[t].distance()).sqrt();

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
            cut_graph[t + query_id]._idx = result[t].idx();
            cut_graph[t + query_id]._distance = result[t].distance();
        }
        if result.len() < range {
            for i in 0..range {
                cut_graph[query_id + i]._distance = E::max_value();
            }
        }
    }

    fn inter_insert(
        &self,
        n: usize,
        range: usize,
        cut_graph: &mut Vec<neighbor::Neighbor<E, usize>>,
    ) {
        for i in 0..range {
            if cut_graph[i + n].distance() == E::from_isize(-1).unwrap() {
                break;
            }

            let sn = neighbor::Neighbor::new(n, cut_graph[i + n].distance());
            let des = cut_graph[i + n].idx();
            let mut temp_pool = Vec::new();
            let mut dup = 0;

            for j in 0..range {
                if cut_graph[j + des].distance() == E::from_isize(-1).unwrap() {
                    break;
                }
                if n == cut_graph[j + des].idx() {
                    dup = 1;
                    break;
                }
                temp_pool.push(cut_graph[j + des].clone());
            }

            if dup == 1 {
                continue;
            }

            temp_pool.push(sn.clone());
            if temp_pool.len() > range {
                let mut result = Vec::new();
                let mut start = 0;
                temp_pool.sort();
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
                        let cos_ij = (p.distance() + result[t].distance() - djk)
                            / E::from_usize(2).unwrap()
                            / (p.distance() * result[t].distance()).sqrt();

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
                    cut_graph[t + des] = result[t].clone();
                }

                if result.len() < range {
                    cut_graph[result.len() + des]._distance = E::from_isize(-1).unwrap();
                }
            } else {
                for t in 0..range {
                    if cut_graph[t + des].distance() == E::from_isize(-1).unwrap() {
                        cut_graph[t + des] = sn.clone();
                        if (t + 1) < range {
                            cut_graph[t + des]._distance = E::from_isize(-1).unwrap();
                            break;
                        }
                    }
                }
            }
        }
    }

    fn build(&mut self) {
        let range = self.index_size;

        self.build_knn_graph();

        let mut cut_graph: Vec<neighbor::Neighbor<E, usize>> =
            Vec::with_capacity(self.nodes.len() * range);
        for i in 0..self.nodes.len() * range {
            cut_graph.push(neighbor::Neighbor::<E, usize>::new(i, E::float_zero()));
            // placeholder
        }
        self.link_each_nodes(&mut cut_graph);

        for i in 0..self.nodes.len() {
            let pool = &cut_graph[i..];
            let mut pool_size = 0;
            for j in 0..range {
                if pool[j].distance() == E::from_isize(-1).unwrap() {
                    break;
                }
                pool_size = j;
            }
            pool_size += 1;
            self.graph[i] = Vec::with_capacity(pool_size);
            for j in 0..pool_size {
                self.graph[i].push(pool[j].idx());
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
        args: &arguments::Args,
    ) -> Vec<(node::Node<E, T>, E)> {
        let mut L = args.uget("search_size").unwrap_or(self.index_size);
        if L < self.root_nodes.len() {
            L = self.root_nodes.len();
        }
        let mut result_set = Vec::new();
        let mut init_ids = vec![0; L];
        let mut flags = HashSet::new();

        self.get_random_nodes_idx(&mut init_ids);
        for i in 0..self.root_nodes.len() {
            init_ids[i] = self.root_nodes[i];
        }

        for i in 0..L {
            let id = init_ids[i];
            let dist = self.nodes[id].metric(query, self.mt).unwrap();
            result_set.push(SubNeighbor::new(id, dist, true));
            flags.insert(id);
        }
        result_set.sort();
        let mut k = 0;
        while k < L {
            let mut nk = L;

            if result_set[k].flag {
                result_set[k].flag = false;
                let n = result_set[k].idx();

                for m in 0..self.graph[n].len() {
                    let id = self.graph[n][m];
                    if flags.contains(&id) {
                        continue;
                    }
                    flags.insert(id);
                    let dist = self.nodes[id].metric(query, self.mt).unwrap();
                    if dist >= result_set[L - 1].distance() {
                        continue;
                    }
                    let nn = SubNeighbor::new(id, dist, true);
                    let r = self.insert_into_pools(&mut result_set, L, &nn);
                    if r < nk {
                        nk = r;
                    }
                }
            }
            if nk <= k {
                k = nk;
            } else {
                k += 1;
            }
        }

        let mut result = Vec::new();
        for i in 0..k {
            result.push((
                *self.nodes[result_set[i].idx()].clone(),
                result_set[i].distance(),
            ));
        }
        result
    }

    fn check_edge(&self, h: usize, t: usize) -> bool {
        let mut flag = true;
        for i in 0..self.graph[h].len() {
            if t == self.graph[h][i] {
                flag = false;
            }
        }
        return flag;
    }

    fn dfs(
        &mut self,
        flags: &mut [bool],
        edges: &mut Vec<(usize, usize)>,
        root: usize,
        cnt: &mut usize,
    ) {
        let mut tmp = root;
        let mut s = Vec::new(); // as stack
        s.push(root.clone());
        if !flags[root] {
            *cnt += 1;
        }
        flags[root] = true;
        while !s.is_empty() {
            let next = self.nodes.len();
            for i in 0..self.graph[tmp].len() {
                if !flags[self.graph[tmp][i]] {
                    let next = self.graph[tmp][i];
                    break;
                }
            }

            if next == self.nodes.len() {
                let head = s.pop().unwrap();
                if s.is_empty() {
                    break;
                }
                tmp = s[s.len() - 1];
                let tail = tmp;
                if self.check_edge(head, tail) {
                    edges.push((head, tail));
                }
                continue;
            }
            tmp = next;
            flags[tmp] = true;
            s.push(tmp.clone());
            *cnt += 1;
        }
    }

    fn find_root(&mut self, flags: &mut [bool], root: &mut usize) {
        let mut id = self.nodes.len();
        for i in 0..self.nodes.len() {
            if !flags[i] {
                id = i;
                break;
            }
        }

        if id == self.nodes.len() {
            return;
        }

        let mut tmp = Vec::new();
        let mut pool = Vec::new();
        self.get_point_neighbors(&self.nodes[id], &mut tmp, &mut pool);

        let mut found = false;
        for i in 0..pool.len() {
            if flags[pool[i].idx()] {
                *root = pool[i].idx();
                found = true;
                break;
            }
        }
        if !found {
            for retry in 0..1000 {
                let rid = rand::thread_rng().gen_range(0, self.nodes.len());
                if flags[rid] {
                    *root = rid;
                    break;
                }
            }
        }
        self.graph[*root].push(id);
    }

    fn strong_connect(&mut self) {
        let n_try = 5;
        let mut edges_all = Vec::new();

        for i in 0..n_try {
            let mut root = rand::thread_rng().gen_range(0, self.nodes.len());
            let mut flags = vec![false; self.nodes.len()];
            let mut unlinked_cnt = 0;
            let mut edges = Vec::new();
            while unlinked_cnt < self.nodes.len() {
                self.dfs(&mut flags, &mut edges, root, &mut unlinked_cnt);
                if unlinked_cnt >= self.nodes.len() {
                    break;
                }
                self.find_root(&mut flags, &mut root);
            }
            for i in 0..edges_all.len() {
                edges_all.push(edges[i]);
            }
        }
    }
}

impl<E: node::FloatElement + DeserializeOwned, T: node::IdxType + DeserializeOwned>
    ann_index::SerializableANNIndex<E, T> for SatelliteSystemGraphIndex<E, T>
{
    fn load(path: &str, args: &arguments::Args) -> Result<Self, &'static str> {
        let mut file = File::open(path).expect(&format!("unable to open file {:?}", path));
        let mut instance: SatelliteSystemGraphIndex<E, T> =
            bincode::deserialize_from(&file).unwrap();
        instance.nodes = instance
            .tmp_nodes
            .iter()
            .map(|x| Box::new(x.clone()))
            .collect();
        Ok(instance)
    }

    fn dump(&mut self, path: &str, args: &arguments::Args) -> Result<(), &'static str> {
        self.tmp_nodes = self.nodes.iter().map(|x| *x.clone()).collect();
        let encoded_bytes = bincode::serialize(&self).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&encoded_bytes)
            .expect(&format!("unable to write file {:?}", path));
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
    fn reconstruct(&mut self, mt: metrics::Metric) {}
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
}
