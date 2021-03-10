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
use serde::de::DeserializeOwned;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::sync::Arc;
use std::thread;
use std::{borrow::Borrow, sync::RwLock};

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct HnswIndex<E: node::FloatElement, T: node::IdxType> {
    _demension: usize, // dimension
    _n_items: usize,   // next item count
    _n_contructed_items: usize,
    _max_item: usize,
    _n_neigh: usize,   // neighbor num except level 0
    _n_neigh0: usize,  // neight num of level 0
    _max_level: usize, //max level
    _cur_level: usize, //current level
    #[serde(skip_serializing, skip_deserializing)]
    _id2neigh: Vec<Vec<RwLock<Vec<usize>>>>, //neight_id from level 1 to level _max_level
    #[serde(skip_serializing, skip_deserializing)]
    _id2neigh0: Vec<RwLock<Vec<usize>>>, //neigh_id at level 0
    #[serde(skip_serializing, skip_deserializing)]
    _datas: Vec<Box<node::Node<E, T>>>, // data saver
    #[serde(skip_serializing, skip_deserializing)]
    _item2id: HashMap<T, usize>, //item_id to id in Hnsw
    _root_id: usize,   //root of hnsw
    _id2level: Vec<usize>,
    _has_deletons: bool,
    _ef_default: usize, // num of max candidates when searching
    #[serde(skip_serializing, skip_deserializing)]
    _delete_ids: HashSet<usize>, //save deleted ids
    _metri: metrics::Metric, //compute metrics

    _id2neigh_tmp: Vec<Vec<Vec<usize>>>,
    _id2neigh0_tmp: Vec<Vec<usize>>,
    _datas_tmp: Vec<node::Node<E, T>>,
    _item2id_tmp: Vec<(T, usize)>,
    _delete_ids_tmp: Vec<usize>,
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
        sorted_list: &Vec<Neighbor<E, usize>>,
        ret_size: usize,
    ) -> Result<Vec<Neighbor<E, usize>>, &'static str> {
        let mut return_list: Vec<Neighbor<E, usize>> = Vec::new();
        let sorted_list_len = sorted_list.len();

        for i in 0..sorted_list_len {
            if return_list.len() >= ret_size {
                break;
            }

            let idx = sorted_list[i].idx();
            let distance = sorted_list[i]._distance;
            if sorted_list_len < ret_size {
                return_list.push(Neighbor::new(idx, distance));
                continue;
            }

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

        return Ok(return_list); // from small to large
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
        sorted_candidates: &Vec<Neighbor<E, usize>>,
        level: usize,
        is_update: bool,
    ) -> Result<usize, &'static str> {
        let n_neigh = if level == 0 {
            self._n_neigh0
        } else {
            self._n_neigh
        };
        let selected_neighbors = self
            .get_neighbors_by_heuristic2(sorted_candidates, n_neigh)
            .unwrap();
        if selected_neighbors.len() > n_neigh {
            return Err("Should be not be more than M_ candidates returned by the heuristic");
        }
        // println!("{:?}",top_candidates);
        if selected_neighbors.len() == 0 {
            return Err("top candidate is empty, impossible!");
        }

        let next_closest_entry_point = selected_neighbors[0].idx();

        {
            let mut cur_neigh = self.get_neighbor(cur_id, level).write().unwrap();
            cur_neigh.clear();
            for i in 0..selected_neighbors.len() {
                cur_neigh.push(selected_neighbors[i].idx());
            }
        }

        for selected_neighbor in selected_neighbors {
            let selected_neighbor_id = selected_neighbor.idx();
            let mut neighbor_of_selected_neighbors = self
                .get_neighbor(selected_neighbor_id, level)
                .write()
                .unwrap();
            if neighbor_of_selected_neighbors.len() > n_neigh {
                return Err("Bad Value of neighbor_of_selected_neighbors");
            }
            if selected_neighbor_id == cur_id {
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
                    let d_max = self.get_distance_from_id(cur_id, selected_neighbor_id);

                    let mut candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
                    candidates.push(Neighbor::new(cur_id, d_max));
                    for i in 0..neighbor_of_selected_neighbors.len() {
                        let neighbor_id = neighbor_of_selected_neighbors[i];
                        let d_neigh = self.get_distance_from_id(neighbor_id, selected_neighbor_id);
                        candidates.push(Neighbor::new(neighbor_id, d_neigh));
                    }
                    let return_list = self
                        .get_neighbors_by_heuristic2(&candidates.into_sorted_vec(), n_neigh)
                        .unwrap();

                    neighbor_of_selected_neighbors.clear();
                    for neighbor_in_list in return_list {
                        neighbor_of_selected_neighbors.push(neighbor_in_list.idx());
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

    pub fn search_layer_with_candidate(
        &self,
        search_data: &node::Node<E, T>,
        sorted_candidates: &Vec<Neighbor<E, usize>>,
        visited_id: &mut HashSet<usize>,
        level: usize,
        ef: usize,
        has_deletion: bool,
    ) -> BinaryHeap<Neighbor<E, usize>> {
        let mut candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        let mut top_candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        for neigh in sorted_candidates {
            let root = neigh.idx();
            if !has_deletion || !self.is_deleted(root) {
                let dist = self.get_distance_from_vec(self.get_data(root), search_data);
                top_candidates.push(Neighbor::new(root, dist));
                candidates.push(Neighbor::new(root, -dist));
            } else {
                candidates.push(Neighbor::new(root, -E::max_value()))
            }
            visited_id.insert(root);
        }
        let mut lower_bound = if top_candidates.is_empty() {
            E::max_value() //max dist in top_candidates
        } else {
            top_candidates.peek().unwrap()._distance
        };

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
                if visited_id.contains(&neigh) {
                    continue;
                }
                visited_id.insert(neigh);
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
    //find ef nearist nodes to search data from root at level
    pub fn search_layer(
        &self,
        root: usize,
        search_data: &node::Node<E, T>,
        level: usize,
        ef: usize,
        has_deletion: bool,
    ) -> BinaryHeap<Neighbor<E, usize>> {
        let mut visited_id: HashSet<usize> = HashSet::new();
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
        visited_id.insert(root);

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
                if visited_id.contains(&neigh) {
                    continue;
                }
                visited_id.insert(neigh);
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

    // pub fn search_layer_default(
    //     &self,
    //     root: usize,
    //     search_data: &node::Node<E, T>,
    //     level: usize,
    // ) -> BinaryHeap<Neighbor<E, usize>> {
    //     return self.search_layer(root, search_data, level, self._ef_default, false);
    // }

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

        top_candidate = self.search_layer(cur_id, search_data, 0, search_range, self._has_deletons);
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
        let neigh0: RwLock<Vec<usize>> = RwLock::new(Vec::with_capacity(self._n_neigh0));
        let mut neigh: Vec<RwLock<Vec<usize>>> = Vec::with_capacity(cur_level);
        for i in 0..cur_level {
            let level_neigh: RwLock<Vec<usize>> = RwLock::new(Vec::with_capacity(self._n_neigh));
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

        let mut level = if insert_level < self._cur_level {
            insert_level
        } else {
            self._cur_level
        };
        let mut visited_id: HashSet<usize> = HashSet::new();
        let mut sorted_candidates: Vec<Neighbor<E, usize>> = Vec::new();
        sorted_candidates.push(Neighbor::new(
            cur_id,
            self.get_distance_from_id(cur_id, insert_id),
        ));
        loop {
            let insert_data = self.get_data(insert_id);
            // let mut visited_id: HashSet<usize> = HashSet::new();
            let mut top_candidates = self.search_layer_with_candidate(
                insert_data,
                &sorted_candidates,
                &mut visited_id,
                level,
                self._ef_default,
                false,
            );
            // let mut top_candidates = self.search_layer_default(cur_id, insert_data, level);
            if self.is_deleted(cur_id) {
                let cur_dist = self.get_distance_from_id(cur_id, insert_id);
                top_candidates.push(Neighbor::new(cur_id, cur_dist));
                if top_candidates.len() > self._ef_default {
                    top_candidates.pop();
                }
            }
            // println!("cur_id {:?}", insert_id);
            // println!("{:?}", top_candidates);
            sorted_candidates = top_candidates.into_sorted_vec();
            if sorted_candidates.is_empty() {
                return Err("sorted sorted_candidate is empty");
            }
            cur_id = self
                .connect_neighbor(insert_id, &sorted_candidates, level, false)
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
        let mut result: Vec<(node::Node<E, T>, E)> = Vec::with_capacity(k);
        let mut result_idx: Vec<(usize, E)> = Vec::with_capacity(k);
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

    fn reconstruct(&mut self, mt: metrics::Metric) {}

    fn name(&self) -> &'static str {
        "HnswIndex"
    }
}

impl<E: node::FloatElement + DeserializeOwned, T: node::IdxType + DeserializeOwned>
    ann_index::SerializableIndex<E, T> for HnswIndex<E, T>
{
    fn load(path: &str, args: &arguments::Args) -> Result<Self, &'static str> {
        let mut file = File::open(path).expect(&format!("unable to open file {:?}", path));
        let mut instance: HnswIndex<E, T> = bincode::deserialize_from(&file).unwrap();
        instance._datas = instance
            ._datas_tmp
            .iter()
            .map(|x| Box::new(x.clone()))
            .collect();
        instance._id2neigh = Vec::with_capacity(instance._id2neigh_tmp.len());
        for i in 0..instance._id2neigh.len() {
            let mut tmp = Vec::with_capacity(instance._id2neigh_tmp[i].len());
            for j in 0..instance._id2neigh_tmp[i].len() {
                tmp.push(RwLock::new(instance._id2neigh_tmp[i][j].clone()));
            }
            instance._id2neigh.push(tmp);
        }
        instance._id2neigh0 = Vec::with_capacity(instance._id2neigh0_tmp.len());
        for i in 0..instance._id2neigh0_tmp.len() {
            instance
                ._id2neigh0
                .push(RwLock::new(instance._id2neigh0_tmp[i].clone()));
        }

        instance._item2id = HashMap::new();
        for iter in instance._item2id_tmp.iter() {
            let (k, v) = &*iter;
            instance._item2id.insert(k.clone(), v.clone());
        }

        instance._delete_ids = HashSet::new();
        for iter in instance._delete_ids_tmp.iter() {
            instance._delete_ids.insert(iter.clone());
        }
        instance._id2neigh_tmp.clear();
        instance._id2neigh0_tmp.clear();
        instance._datas_tmp.clear();
        instance._item2id_tmp.clear();
        instance._delete_ids_tmp.clear();
        Ok(instance)
    }

    fn dump(&mut self, path: &str, args: &arguments::Args) -> Result<(), &'static str> {
        self._id2neigh_tmp = Vec::with_capacity(self._id2neigh.len());
        for i in 0..self._id2neigh.len() {
            let mut tmp = Vec::with_capacity(self._id2neigh[i].len());
            for j in 0..self._id2neigh[i].len() {
                tmp.push(self._id2neigh[i][j].read().unwrap().clone());
            }
            self._id2neigh_tmp.push(tmp);
        }

        self._id2neigh0_tmp = Vec::with_capacity(self._id2neigh0.len());
        for i in 0..self._id2neigh0.len() {
            self._id2neigh0_tmp
                .push(self._id2neigh0[i].read().unwrap().clone());
        }

        self._datas_tmp = self._datas.iter().map(|x| *x.clone()).collect();
        self._item2id_tmp = Vec::with_capacity(self._item2id.len());
        for (k, v) in &self._item2id {
            self._item2id_tmp.push((k.clone(), v.clone()));
        }
        self._delete_ids_tmp = Vec::new();
        for iter in &self._delete_ids {
            self._delete_ids_tmp.push(iter.clone());
        }

        let encoded_bytes = bincode::serialize(&self).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&encoded_bytes)
            .expect(&format!("unable to write file {:?}", path));
        Result::Ok(())
    }
}
