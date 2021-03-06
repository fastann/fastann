use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor::Neighbor;
use crate::core::node;
use ann_index::ANNIndex;
use hashbrown::HashMap;
use hashbrown::HashSet;
use rand::prelude::*;
use rayon::{iter::IntoParallelIterator, prelude::*};
use std::sync::Arc;
use std::sync::RwLock;
use std::thread;
#[derive(Default, Debug)]
pub struct HnswIndex<E: node::FloatElement, T: node::IdxType> {
    _demension: usize, // dimension
    _n_items: usize,   // next item count
    _n_contructed_items: usize,
    _max_item: usize,
    _n_neigh: usize,                         // neighbor num except level 0
    _n_neigh0: usize,                        // neight num of level 0
    _max_level: usize,                       //max level
    _cur_level: usize,                       //current level
    _id2neigh: Vec<Vec<RwLock<Vec<usize>>>>, //neight_id from level 1 to level _max_level
    _id2neigh0: Vec<RwLock<Vec<usize>>>,     //neigh_id at level 0
    _datas: Vec<Box<node::Node<E, T>>>,      // data saver
    _item2id: HashMap<T, usize>,             //item_id to id in Hnsw
    _root_id: usize,                         //root of hnsw
    _id2level: Vec<usize>,
    _has_deletons: bool,
    _ef_default: usize,          // num of max candidates when searching
    _delete_ids: HashSet<usize>, //save deleted ids
    _metri: metrics::Metric,     //compute metrics
}

impl<E: node::FloatElement, T: node::IdxType> HnswIndex<E, T> {
    pub fn new(
        demension: usize,
        max_item: usize,
        n_neigh: usize,
        n_neigh0: usize,
        max_level: usize,
        ef: usize,
        has_deletion: bool,
    ) -> HnswIndex<E, T> {
        return HnswIndex {
            _demension: demension,
            _n_items: 0,
            _n_contructed_items: 0,
            _max_item: max_item,
            _n_neigh: n_neigh,
            _n_neigh0: n_neigh0,
            _max_level: max_level,
            _cur_level: 0,
            _root_id: 0,
            _has_deletons: has_deletion,
            _ef_default: ef,
            _metri: metrics::Metric::Manhattan,
            ..Default::default()
        };
    }

    fn get_random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut ret = 0;
        while ret < self._max_level {
            if rng.gen_range(0.0, 1.0) > 0.5 {
                ret += 1;
            } else {
                break;
            }
        }
        return ret;
    }
    //input top_candidate as max top heap
    //return min top heap in top_candidates, delete part candidate
    fn get_neighbors_by_heuristic2(
        &self,
        top_candidates: &mut BinaryHeap<Neighbor<E, usize>>,
        ret_size: usize,
    ) -> Result<(), &'static str> {
        if top_candidates.len() <= ret_size {
            return Ok(());
        }
        let mut queue_closest: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        let mut return_list: Vec<Neighbor<E, usize>> = Vec::new();
        while !top_candidates.is_empty() {
            let cand = top_candidates.peek().unwrap();
            queue_closest.push(Neighbor::new(cand.idx(), -cand._distance));
            top_candidates.pop();
        }

        while !queue_closest.is_empty() {
            if return_list.len() >= ret_size {
                break;
            }
            let cur = queue_closest.peek().unwrap();
            let idx = cur.idx();
            let distance = -cur._distance;
            queue_closest.pop();
            let mut good = true;

            for ret_neighbor in &return_list {
                let cur2ret_dis = self.get_distance_from_id(idx, ret_neighbor.idx());
                if cur2ret_dis < distance {
                    good = false;
                    break;
                }
            }

            if good {
                return_list.push(Neighbor::new(idx, distance));
            }
        }

        for ret_neighbor in &return_list {
            top_candidates.push(Neighbor::new(ret_neighbor.idx(), ret_neighbor._distance));
        }

        return Ok(());
    }

    fn get_neighbor(&self, id: usize, level: usize) -> &RwLock<Vec<usize>> {
        if level == 0 {
            return &self._id2neigh0[id];
        }
        return &self._id2neigh[id][level - 1];
    }

    #[allow(dead_code)]
    fn get_level(&self, id: usize) -> usize {
        return self._id2level[id];
    }

    fn connect_neighbor(
        &self,
        cur_id: usize,
        top_candidates: &mut BinaryHeap<Neighbor<E, usize>>,
        level: usize,
        is_update: bool,
    ) -> Result<usize, &'static str> {
        let n_neigh = if level == 0 {
            self._n_neigh0
        } else {
            self._n_neigh
        };
        self.get_neighbors_by_heuristic2(top_candidates, n_neigh)
            .unwrap();
        if top_candidates.len() > n_neigh {
            return Err("Should be not be more than M_ candidates returned by the heuristic");
        }
        // println!("{:?}",top_candidates);

        let mut selected_neighbors: Vec<usize> = Vec::new();
        while !top_candidates.is_empty() {
            // can remove for efficience
            selected_neighbors.push(top_candidates.peek().unwrap().idx());
            top_candidates.pop();
        }

        let next_closest_entry_point = selected_neighbors[0];

        {
            let mut cur_neigh = self.get_neighbor(cur_id, level).write().unwrap();
            cur_neigh.clear();
            for i in 0..selected_neighbors.len() {
                cur_neigh.push(selected_neighbors[i]);
            }
        }

        for selected_neighbor in selected_neighbors {
            let mut neighbor_of_selected_neighbors =
                self.get_neighbor(selected_neighbor, level).write().unwrap();
            if neighbor_of_selected_neighbors.len() > n_neigh {
                return Err("Bad Value of neighbor_of_selected_neighbors");
            }
            if selected_neighbor == cur_id {
                return Err("Trying to connect an element to itself");
            }

            let mut is_cur_id_present = false;

            if is_update {
                for j in 0..neighbor_of_selected_neighbors.len() {
                    if neighbor_of_selected_neighbors[j] == cur_id {
                        is_cur_id_present = true;
                        break;
                    }
                }
            }

            if !is_cur_id_present {
                if neighbor_of_selected_neighbors.len() < n_neigh {
                    neighbor_of_selected_neighbors.push(cur_id);
                } else {
                    let d_max = self.get_distance_from_id(cur_id, selected_neighbor);

                    let mut candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
                    candidates.push(Neighbor::new(cur_id, d_max));
                    for i in 0..neighbor_of_selected_neighbors.len() {
                        let neighbor_id = neighbor_of_selected_neighbors[i];
                        let d_neigh = self.get_distance_from_id(neighbor_id, selected_neighbor);
                        candidates.push(Neighbor::new(neighbor_id, d_neigh));
                    }
                    self.get_neighbors_by_heuristic2(&mut candidates, n_neigh)
                        .unwrap();

                    neighbor_of_selected_neighbors.clear();
                    while !candidates.is_empty() {
                        // selected_neighbor = candidates.peek().unwrap().idx();
                        neighbor_of_selected_neighbors.push(candidates.peek().unwrap().idx());
                        candidates.pop();
                    }
                }
            }
        }

        return Ok(next_closest_entry_point);
    }

    #[allow(dead_code)]
    pub fn delete_id(&mut self, id: usize) -> Result<(), &'static str> {
        if id > self._n_contructed_items {
            return Err("Invalid delete id");
        }
        if self.is_deleted(id) {
            return Err("id has deleted");
        }
        self._delete_ids.insert(id);
        return Ok(());
    }

    pub fn is_deleted(&self, id: usize) -> bool {
        return self._has_deletons && self._delete_ids.contains(&id);
    }

    pub fn get_data(&self, id: usize) -> &node::Node<E, T> {
        return &self._datas[id];
    }

    pub fn get_distance_from_vec(&self, x: &node::Node<E, T>, y: &node::Node<E, T>) -> E {
        return metrics::metric(x.vectors(), y.vectors(), self._metri).unwrap();
    }

    pub fn get_distance_from_id(&self, x: usize, y: usize) -> E {
        return metrics::metric(
            self.get_data(x).vectors(),
            self.get_data(y).vectors(),
            self._metri,
        )
        .unwrap();
    }

    //find ef nearist nodes to search data from root at level
    pub fn search_laryer(
        &self,
        root: usize,
        search_data: &node::Node<E, T>,
        level: usize,
        ef: usize,
        has_deletion: bool,
    ) -> BinaryHeap<Neighbor<E, usize>> {
        let mut visted_id: HashSet<usize> = HashSet::new();
        let mut top_candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        let mut candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        let mut lower_bound: E;

        if !has_deletion || !self.is_deleted(root) {
            let dist = self.get_distance_from_vec(self.get_data(root), search_data);
            top_candidates.push(Neighbor::new(root, dist));
            candidates.push(Neighbor::new(root, -dist));
            lower_bound = dist;
        } else {
            lower_bound = E::max_value(); //max dist in top_candidates
            candidates.push(Neighbor::new(root, -lower_bound))
        }
        visted_id.insert(root);

        while !candidates.is_empty() {
            let cur_neigh = candidates.peek().unwrap();
            let cur_dist = -cur_neigh._distance;
            let cur_id = cur_neigh.idx();
            candidates.pop();
            if cur_dist > lower_bound {
                break;
            }
            let cur_neighbors = self.get_neighbor(cur_id, level).read().unwrap();
            for i in 0..cur_neighbors.len() {
                let neigh = cur_neighbors[i];
                if visted_id.contains(&neigh) {
                    continue;
                }
                visted_id.insert(neigh);
                let dist = self.get_distance_from_vec(self.get_data(neigh), search_data);
                if top_candidates.len() < ef || dist < lower_bound {
                    candidates.push(Neighbor::new(neigh, -dist));

                    if !self.is_deleted(neigh) {
                        top_candidates.push(Neighbor::new(neigh, dist))
                    }

                    if top_candidates.len() > ef {
                        top_candidates.pop();
                    }

                    if !top_candidates.is_empty() {
                        lower_bound = top_candidates.peek().unwrap()._distance;
                    }
                }
            }
        }

        return top_candidates;
    }

    pub fn search_laryer_default(
        &self,
        root: usize,
        search_data: &node::Node<E, T>,
        level: usize,
    ) -> BinaryHeap<Neighbor<E, usize>> {
        return self.search_laryer(root, search_data, level, self._ef_default, false);
    }

    pub fn search_knn(
        &self,
        search_data: &node::Node<E, T>,
        k: usize,
    ) -> Result<BinaryHeap<Neighbor<E, usize>>, &'static str> {
        let mut top_candidate: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        if self._n_contructed_items == 0 {
            return Ok(top_candidate);
        }
        let mut cur_id = self._root_id;
        let mut cur_dist = self.get_distance_from_vec(self.get_data(cur_id), search_data);
        let mut cur_level = self._cur_level;
        loop {
            let mut changed = true;
            while changed {
                changed = false;
                let cur_neighs = self
                    .get_neighbor(cur_id, cur_level as usize)
                    .read()
                    .unwrap();
                for i in 0..cur_neighs.len() {
                    let neigh = cur_neighs[i];
                    if neigh > self._max_item {
                        return Err("cand error");
                    }
                    let dist = self.get_distance_from_vec(self.get_data(cur_id), search_data);
                    if dist < cur_dist {
                        cur_dist = dist;
                        cur_id = neigh;
                        changed = true;
                    }
                }
            }
            if cur_level == 0 {
                break;
            }
            cur_level -= 1;
        }

        let search_range = if self._ef_default > k {
            self._ef_default
        } else {
            k
        };

        top_candidate =
            self.search_laryer(cur_id, search_data, 0, search_range, self._has_deletons);
        while top_candidate.len() > k {
            top_candidate.pop();
        }

        return Ok(top_candidate);
    }

    pub fn init_item(&mut self, data: &node::Node<E, T>) -> usize {
        let cur_id = self._n_items;
        let mut cur_level = self.get_random_level();
        if cur_id == 0 {
            cur_level = self._max_level;
            self._cur_level = cur_level;
            self._root_id = cur_id;
        }
        let neigh0: RwLock<Vec<usize>> = RwLock::new(Vec::new());
        let mut neigh: Vec<RwLock<Vec<usize>>> = Vec::new();
        for i in 0..cur_level {
            let level_neigh: RwLock<Vec<usize>> = RwLock::new(Vec::new());
            neigh.push(level_neigh);
        }
        self._datas.push(Box::new(data.clone()));
        self._id2neigh0.push(neigh0);
        self._id2neigh.push(neigh);
        self._id2level.push(cur_level);
        // self._item2id.insert(data.idx().unwrap(), cur_id);
        self._n_items += 1;
        return cur_id;
    }

    fn batch_construct(&mut self, mt: metrics::Metric) -> Result<(), &'static str> {
        if self._n_items < self._n_contructed_items {
            return Err("contruct error");
        }

        (self._n_contructed_items..self._n_items)
            .into_par_iter()
            .for_each(|insert_id: usize| {
                self.construct_single_item(insert_id);
                // println!("insert_id {}", insert_id);
                return;
            });

        // for insert_id in self._n_contructed_items..self._n_items{
        //     // println!("insert id {}", insert_id);
        //     self.construct_single_item(insert_id);
        // }

        self._n_contructed_items = self._n_items;
        return Ok(());
    }

    pub fn add_item_not_constructed(
        &mut self,
        data: &node::Node<E, T>,
    ) -> Result<(), &'static str> {
        if data.len() != self._demension {
            return Err("dimension is different");
        }
        {
            // if self._item2id.contains_key(data.idx().unwrap()) {
            //     //to_do update point
            //     return Ok(self._item2id[data.idx().unwrap()]);
            // }

            if self._n_items >= self._max_item {
                return Err("The number of elements exceeds the specified limit");
            }
        }

        let insert_id = self.init_item(data);
        let insert_level = self.get_level(insert_id);
        return Ok(());
    }

    pub fn add_single_item(&mut self, data: &node::Node<E, T>) -> Result<(), &'static str> {
        //not support asysn
        if data.len() != self._demension {
            return Err("dimension is different");
        }
        {
            // if self._item2id.contains_key(data.idx().unwrap()) {
            //     //to_do update point
            //     return Ok(self._item2id[data.idx().unwrap()]);
            // }

            if self._n_items >= self._max_item {
                return Err("The number of elements exceeds the specified limit");
            }
        }

        let insert_id = self.init_item(data);
        let insert_level = self.get_level(insert_id);
        self.construct_single_item(insert_id);

        self._n_contructed_items += 1;

        return Ok(());
    }

    pub fn construct_single_item(&self, insert_id: usize) -> Result<(), &'static str> {
        let insert_level = self._id2level[insert_id];
        // println!("insert id {} insert_level {}", insert_id, insert_level);
        // println!("self._cur_level {}", self._cur_level);
        let mut cur_id = self._root_id;
        // println!("insert_id {:?}, insert_level {:?} ", insert_id, insert_level);

        if insert_id == 0 {
            return Ok(());
        }

        if insert_level < self._cur_level {
            let mut cur_dist = self.get_distance_from_id(cur_id, insert_id);
            let mut cur_level = self._cur_level;
            while cur_level > insert_level {
                let mut changed = true;
                while changed {
                    changed = false;
                    let cur_neighs = self.get_neighbor(cur_id, cur_level).read().unwrap();
                    for i in 0..cur_neighs.len() {
                        let cur_neigh = cur_neighs[i];
                        if cur_neigh > self._n_items {
                            return Err("cand error");
                        }
                        let neigh_dist = self.get_distance_from_id(cur_neigh, insert_id);
                        if neigh_dist < cur_dist {
                            cur_dist = neigh_dist;
                            cur_id = cur_neigh;
                            changed = true;
                        }
                    }
                }
                cur_level -= 1;
            }
        }

        let is_deleted = self.is_deleted(cur_id);
        let mut level = if insert_level < self._cur_level {
            insert_level
        } else {
            self._cur_level
        };
        loop {
            let insert_data = self.get_data(insert_id);
            let mut top_candidates = self.search_laryer_default(cur_id, insert_data, level);
            if is_deleted {
                let cur_dist = self.get_distance_from_id(cur_id, insert_id);
                top_candidates.push(Neighbor::new(cur_id, cur_dist));
                if top_candidates.len() > self._ef_default {
                    top_candidates.pop();
                }
            }
            // println!("cur_id {:?}", insert_id);
            // println!("{:?}", top_candidates);
            cur_id = self
                .connect_neighbor(insert_id, &mut top_candidates, level, false)
                .unwrap();

            if level == 0 {
                break;
            }
            level -= 1;
        }
        return Ok(());
    }
}

impl<E: node::FloatElement, T: node::IdxType> ann_index::ANNIndex<E, T> for HnswIndex<E, T> {
    fn construct(&mut self, mt: metrics::Metric) -> Result<(), &'static str> {
        return self.batch_construct(mt);
    }
    fn add_node(&mut self, item: &node::Node<E, T>) -> Result<(), &'static str> {
        self.add_item_not_constructed(item)
    }
    fn once_constructed(&self) -> bool {
        true
    }

    fn node_search_k(
        &self,
        item: &node::Node<E, T>,
        k: usize,
        args: &arguments::Args,
    ) -> Vec<(node::Node<E, T>, E)> {
        let mut ret: BinaryHeap<Neighbor<E, usize>> = self.search_knn(item, k).unwrap();
        let mut result: Vec<(node::Node<E, T>, E)> = Vec::new();
        let mut result_idx: Vec<(usize, E)> = Vec::new();
        while !ret.is_empty() {
            let top = ret.peek().unwrap();
            let top_idx = top.idx();
            let top_distance = top.distance();
            ret.pop();
            result_idx.push((top_idx, top_distance))
        }
        for i in 0..result_idx.len() {
            let cur_id = result_idx.len() - i - 1;
            result.push((
                *self._datas[result_idx[cur_id].0].clone(),
                result_idx[cur_id].1,
            ));
        }
        return result;
    }

    fn load(&self, path: &str) -> Result<(), &'static str> {
        Result::Ok(())
    }

    fn dump(&self, path: &str) -> Result<(), &'static str> {
        Result::Ok(())
    }

    fn reconstruct(&mut self, mt: metrics::Metric) {}

    fn name(&self) -> &'static str {
        "HnswIndex"
    }
}
