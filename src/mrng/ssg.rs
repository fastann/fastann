use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use rand::prelude::*;
extern crate num;
use bincode;
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::cmp;
use std::collections::HashSet;
use std::collections::LinkedList;
use std::collections::VecDeque;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
            threshold: (angle / E::from_f32(180.0).unwrap() * E::PI()).cos(),
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

        let mut tmp: Vec<neighbor::Neighbor<E, usize>> = Vec::new();
        let mut pool: Vec<neighbor::Neighbor<E, usize>> = Vec::new();
        let n = node::Node::new(&center);
        self.get_point_neighbors(&n, &mut tmp, &mut pool);
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
        addr: &mut Vec<neighbor::Neighbor<E, usize>>,
        k: usize,
        nn: &neighbor::Neighbor<E, usize>,
    ) -> usize {
        addr.push(nn.clone());
        addr.sort();
        for i in 0..cmp::max(addr.len(), k) {
            if addr[i].idx() == nn.idx() {
                return i;
            }
        }
        let mut left = 0;
        let mut right = k - 1;
        if addr[left].distance() < nn.distance() {
            addr[left] = nn.clone();
            return left;
        }
        if addr[right].distance() > nn.distance() {
            addr[k] = nn.clone();
            return k;
        }
        while left < right - 1 {
            let mid = (left + right) / 2;
            if addr[mid].distance() < nn.distance() {
                right = mid;
            } else {
                left = mid;
            }
        }
        while left > 0 {
            if addr[left].distance() > nn.distance() {
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
        return_set: &mut Vec<neighbor::Neighbor<E, usize>>,
        full_set: &mut Vec<neighbor::Neighbor<E, usize>>,
    ) {
        let mut init_ids = vec![0; self.L];
        self.get_random_nodes_idx(&mut init_ids);
        let mut flags = vec![false; self.nodes.len()]; // as bitmap;
        let mut _L = 0;
        let mut return_set_flags = vec![false; self.nodes.len()];
        for i in 0..init_ids.len() + 1 {
            let id = init_ids[i];
            if id > self.nodes.len() {
                continue;
            }
            return_set.push(neighbor::Neighbor::new(
                id.clone(),
                item.metric(&self.nodes[id], self.mt).unwrap(),
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
                    full_set.push(neighbor::Neighbor::new(id, dist));
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

    fn get_point_L_neighbors(&self, q: usize, pool: &mut Vec<neighbor::Neighbor<E, usize>>) {
        let mut flags = HashSet::new();

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
                if pool.len() >= self.L {
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
        (0..self.nodes.len()).for_each(|i| {
            pool.clear();
            self.get_point_L_neighbors(i, &mut pool); // get related one
            self.prune_graph(i, &mut pool, threshold, cut_graph);
        });
        cut_graph.iter().for_each(|x| {
            if x.idx() > self.nodes.len() + 1 {
                println!("here {:?}", x);
            }
        });
        (0..self.nodes.len()).for_each(|i| {
            self.inter_insert(i, range, cut_graph);
        });

        Result::Ok(())
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
        let mut flags = HashSet::new();
        for i in 0..related_ids_pool.len() {
            flags.insert(related_ids_pool[i].idx());
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

        related_ids_pool.sort();
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
            for t in 0..result.len() {
                if p.idx() == result[t].idx() {
                    // stop early
                    occlude = true;
                    break;
                }
                let djk = self.nodes[result[t].idx()]
                    .metric(&self.nodes[p.idx()], self.mt)
                    .unwrap();
                let cos_ij = (p.distance().powi(2) + result[t].distance().powi(2) - djk.powi(2))
                    / (E::from_usize(2).unwrap() * (p.distance() * result[t].distance()));

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
                cut_graph[query_id * self.index_size + i]._idx = self.nodes.len(); // means not exist
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
            Vec::with_capacity(self.nodes.len() *  self.index_size);
        for i in 0..self.nodes.len() *  self.index_size {
            cut_graph.push(neighbor::Neighbor::<E, usize>::new(i, E::float_zero()));
        }
        self.link_each_nodes(&mut cut_graph);

        for i in 0..self.nodes.len() {
            let mut pool_size = 0;
            for j in 0.. self.index_size {
                if cut_graph[i*self.index_size + j].distance() == E::max_value() {
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
        args: &arguments::Args,
    ) -> Vec<(node::Node<E, T>, E)> {
        let mut L = k;
        if L < self.root_nodes.len() {
            L = self.root_nodes.len();
        }
        println!("{:?}", L);
        let mut init_ids = vec![0; L];
        let mut search_flags = HashSet::with_capacity(self.nodes.len());
        let mut heap: BinaryHeap<neighbor::Neighbor<E, usize>> = BinaryHeap::new(); // max-heap
        let mut search_queue = LinkedList::new();

        self.get_random_nodes_idx(&mut init_ids);
        for i in 0..self.root_nodes.len() {
            init_ids[i] = self.root_nodes[i];
        }

        for i in 0..L {
            let id = init_ids[i];
            let dist = self.nodes[id].metric(query, self.mt).unwrap();
            heap.push(neighbor::Neighbor::new(id, dist));
            for j in 0..self.graph[id].len() {
                search_queue.push_back(self.graph[id][j]);
            }
            search_flags.insert(id);
        }

        let mut search_times = 0;
        while !search_queue.is_empty() {
            let id = search_queue.pop_front().unwrap();
            if search_flags.contains(&id) {
                continue;
            }

            let dist = self.nodes[id].metric(query, self.mt).unwrap();
            if dist < heap.peek().unwrap().distance() {
                heap.pop();
                heap.push(neighbor::Neighbor::new(id, dist));
                for j in 0..self.graph[id].len() {
                    search_queue.push_back(self.graph[id][j]);
                }
            }
            search_flags.insert(id);
            search_times += 1;
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
                queue.push_back(self.graph[id][x].clone());
                if self.graph[id][x] > self.nodes.len() {
                    // println!("{:?} {:?} {:?}", self.graph[id][x], self.graph[id], id);
                }
            }
            visited.insert(id.clone());
        }

        println!("connectivity: {:?} {:?}", self.nodes.len(), visited.len());
    }
}

impl<E: node::FloatElement + DeserializeOwned, T: node::IdxType + DeserializeOwned>
    ann_index::SerializableIndex<E, T> for SatelliteSystemGraphIndex<E, T>
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

    fn nodes_size(&self) -> usize {
        self.nodes.len()
    }
}
