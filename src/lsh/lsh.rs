use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor::Neighbor;
use crate::core::node;
use metrics::metric;
use rand::prelude::*;
use serde::de::DeserializeOwned;

use serde::{Deserialize, Serialize};

use std::{fs::File, process::exit, usize};

use std::io::Write;


static hamdis_tab_ham_bytes:[u8;256] = [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
    2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4,
    2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8];

#[derive(Default, Debug)]
pub struct LSHIndex<E: node::FloatElement, T: node::IdxType> {
    _dimension: usize,     //dimension of data
    _nbit: usize, // nb of bits per vector
    _n_items: usize,
    _n_constructed_items: usize,
    _max_item: usize,
    _bytes_per_vec: usize, // nb of 8-bits per encoded vector
    // _rotate_data: bool, //< whether to apply a random rotation to input
    _has_median: bool, //< whether we train thresholds or use 0
    _median: Vec<E>, // median data
    _bit_code: Vec<Vec<u8>>, //saved data code
    _nodes: Vec<Box<node::Node<E, T>>>,
    mt: metrics::Metric, //compute metrics
}

impl<E: node::FloatElement, T: node::IdxType> LSHIndex<E, T> {
    pub fn new(
        dimension: usize,
        nbit: usize,
        max_item: usize,
    ) -> LSHIndex<E, T> {
        LSHIndex{
            _dimension: dimension,
            _n_items: 0,
            _n_constructed_items: 0,
            _max_item: max_item,
            _nbit: nbit,
            _has_median: false,
            mt: metrics::Metric::Unknown,
            ..Default::default()
        }
    }

    pub fn init_item(&mut self, data: &node::Node<E, T>) -> usize {
        let cur_id = self._n_items;
        // self._item2id.insert(item, cur_id);
        self._nodes.push(Box::new(data.clone()));
        self._n_items += 1;
        cur_id
    }

    pub fn add_item_not_constructed(&mut self, data: &node::Node<E, T>) -> Result<usize, &'static str> {
        if data.len() != self._dimension {
            return Err("dimension is different");
        }

        if self._n_items > self._max_item {
            return Err("The number of elements exceeds the specified limit");
        }

        let insert_id = self.init_item(data);
        Ok(insert_id)
    }

    pub fn batch_construct(&mut self){
        if self._has_median{
            let d = self._dimension;
            self._median.resize(d, E::zero());
            for i in 0..d {
                let mut sort_data = Vec::<E>::new();
                for j in 0..self._n_items{
                    sort_data.push(self._nodes[j].vectors()[i]);
                }
                sort_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
                self._median[i] = if self._n_items % 2 == 0 {
                    (sort_data[self._n_items/2] + sort_data[self._n_items/2+1])/E::from_f32(2.0).unwrap()
                }
                else{
                    sort_data[self._n_items/2+1]
                };
            }
        }

        for i in self._n_constructed_items..self._n_items{
            let node_data = self._nodes[i].vectors();
            let bit_vec = self.float2bit(node_data);
            // println!("{:?}", bit_vec);
            // println!("{:?}", node_data);
            // println!("{:?}", hamdis_tab_ham_bytes[95]);
            // exit(1);
            self._bit_code.push(bit_vec);

        }
        self._n_constructed_items = self._n_items;
    }

    pub fn float2bit(&self, f_vec: &Vec<E>) -> Vec<u8>{
        let mut bit_vec: Vec<u8> = Vec::new();
        bit_vec.resize(self._nbit, 0);
        let mut i = 0;
        let mut bit_i = 0;
        let d = self._dimension;
        while i < d {
            let mut w: u8 = 0;
            let mut mask: u8 = 1;
            let nj =  if i + 8 <= d { 8 } else { d - i };
            for j in 0..nj {
                let cur = if self._has_median {
                    f_vec[i + j] - self._median[i + j]
                }
                else{
                    f_vec[i + j]
                };
                if cur >= E::from_f32(0.0).unwrap() {
                    w |= mask;
                }
                mask <<= 1;
            }
            bit_vec[bit_i] = w;
            bit_i += 1;
            if bit_i == self._nbit{
                break;
            }
            i += 8;
        }
        return bit_vec;
    }

    pub fn bit2float(&self, mut f_vec: Vec<E>, bit_vec: Vec<u8>){
        for i in 0..self._dimension{
           f_vec[i] = E::from_f32( 
               2.0 * ( ( ( bit_vec[i>>3] >> (i & 7) ) & 1 ) as f32) - 1.0 ).unwrap(); 
        }
    }

    pub fn search_knn(
        &self,
        search_data: &node::Node<E, T>,
        k: usize,
    ) -> Result<BinaryHeap<Neighbor<E, usize>>, &'static str> {
        let mut top_candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        if self._n_constructed_items == 0 {
            return Ok(top_candidates);
        }
        let q_code = self.float2bit(search_data.vectors());

        let n_item = self._n_constructed_items;
        for i in 0..n_item{
            let code = &self._bit_code[i];
            let mut dis = 0.0;
            for i in 0..self._nbit{
                dis += hamdis_tab_ham_bytes[(q_code[i]^code[i]) as usize] as f32;
            }
            top_candidates.push(Neighbor::new(i, E::from_f32(-dis).unwrap()));
            if top_candidates.len() > k {
                top_candidates.pop();
            }
        }
        return Ok(top_candidates);
    }
}

impl<E: node::FloatElement, T: node::IdxType> ann_index::ANNIndex<E, T> for LSHIndex<E, T> {
    fn construct(&mut self, _mt: metrics::Metric) -> Result<(), &'static str> {
        self.mt = _mt;
        self.batch_construct();
        Result::Ok(())
    }
    fn add_node(&mut self, item: &node::Node<E, T>) -> Result<(), &'static str> {
        match self.add_item_not_constructed(item) {
            Err(err) => Err(err),
            _ => Ok(()),
        }
    }
    fn once_constructed(&self) -> bool {
        true
    }

    fn node_search_k(
        &self,
        item: &node::Node<E, T>,
        k: usize,
        _args: &arguments::Args,
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
                *self._nodes[result_idx[cur_id].0].clone(),
                result_idx[cur_id].1,
            ));
        }
        result
    }

    fn reconstruct(&mut self, _mt: metrics::Metric) {}

    fn name(&self) -> &'static str {
        "LSHIndex"
    }
}
