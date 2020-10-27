use crate::common::calc;
use crate::common::neighbor::Neighbor;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use rand::prelude::*;

#[derive(Default, Clone, PartialEq, Debug)]
pub struct Data {
    _demension: i32,
    _val: Vec<f64>
}

impl Data {
    fn new() -> Data {
        Data {
            _demension: 0,
            ..Default::default()
        }
    }

    fn new_with_vectors(demension: i32, val: &[f64]) -> Data {
        Data {
            _demension: demension,
            _val: val.to_vec(),
        }
    }
    
    fn distance(&self, data: &Data) -> Result<f64, &str> {
        let ret = calc::enclidean_distance(&(self._val), &(data._val));
        ret
    }
}

pub struct HnswIndexer{
    _demension: usize, // dimension
    _n_items: i32, // next item count
    _n_level: i32, // cur level 
    _n_neigh: i32, // neighbor num except level 0
    _n_neigh0: i32, // neight num of level 0
    _max_level: i32, //max level
    _cur_level: i32, //current level
    _neigh_id: Vec<Vec<Vec<i32>>>, //neight_id from level 1 to level _max_level
    _neigh0_id: Vec<Vec<i32>>, //neigh_id at level 0
    _datas: Vec<Vec<f64>>, // data saver
    _item2id: HashMap<i32, i32>, //item_id to id in Hnsw
    _root_id: i32 //root of hnsw
}



impl HnswIndexer{
    fn get_random_level(&self) -> i32{
        let mut rng = rand::thread_rng();
        let mut ret = 0;
        while(ret < self._max_level)
        {
            if rng.gen_range(0.0, 1.0) > 0.5 {
                ret += 1;
            }
            else{
                break;
            }
        }
        if ret > self._cur_level {
            ret = self._cur_level + 1
        }
        return ret;
    }

    fn get_distance_from_id(&self, x: i32, y:i32) -> Result<f64, &'static str>{
        return Ok(0.0);
    }
    //input top_candidate as max top heap 
    //return min top heap in top_candidates, delete part candidate
    fn get_neighbors_by_heuristic2(&self, top_candidates: &mut BinaryHeap<Neighbor>, ret_size: i32) -> Result<(), &'static str> {
        if top_candidates.len() < ret_size as usize {
            return Ok(());
        }
        let mut queue_closest: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut return_list : Vec<Neighbor> = Vec::new();
        while !top_candidates.is_empty() {
            let cand = top_candidates.peek().unwrap();
            queue_closest.push(Neighbor::new(cand._idx, -cand._distance));
            top_candidates.pop();
        }

        while !queue_closest.is_empty() {
            if return_list.len() >= ret_size as usize{
                break;
            }
            let cur = queue_closest.peek().unwrap();
            let idx = cur._idx;
            let distance = -cur._distance;
            queue_closest.pop();
            let mut good = true;
            
            for ret_neighbor in &return_list {
                let cur2ret_dis = self.get_distance_from_id(idx, ret_neighbor._idx).unwrap();
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
            top_candidates.push(Neighbor::new(ret_neighbor._idx, -ret_neighbor._distance));
        }

        return Ok(());
    }

    fn connect_neighbor(&self, cur_id: i32, top_candidates: &mut BinaryHeap<Neighbor>, level: i32, bool is_update) -> Result<i32, &'static str>
    {
        let n_neigh = if level == 0 {self._n_neigh0} else {self._n_neigh};
        self.get_neighbors_by_heuristic2(top_candidates, n_neigh);
        if top_candidates.len() > n_neigh as usize {
            return Err("Should be not be more than M_ candidates returned by the heuristic");
        }
        
        let selected_neighbor: Vec<i32> = Vec::new();
        while !top_candidates.is_empty() { // can remove for efficience
            selected_neighbor.push(top_candidates.peek().unwrap()._idx);
            top_candidates.pop();
        }
        
        let next_closest_entry_point = selected_neighbor[0];
        
        

        return Ok(next_closest_entry_point);
    }

    pub fn add_item(&mut self, item: i32, val: &[f64]) -> Result<(), &'static str> {
        if val.len() != self._demension {
            return Err("dimension is different");
        }
        let cur_id = self._n_items;
        self._item2id.insert(item, self._n_items);
        self._datas.push(val.to_vec() );
        self._n_items += 1;

        let mut cur_level = self.get_random_level();
        
        
        if cur_level==0 {
            self._neigh_id.push(vec![].to_vec());
        }
        else{
            let mut neigh_id = vec![];
            for i in 0..cur_level {
                neigh_id.push(vec![]);
            }
            self._neigh_id.push(neigh_id);
        }
        self._neigh0_id.push(vec![]);

        if self._n_items==1 {
            self._root_id = cur_id;
            return Ok(());
        }

        
        
        return Ok(());
    } 
}