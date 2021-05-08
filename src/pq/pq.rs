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

use std::fs::File;

use std::io::Write;

#[derive(Default, Debug)]
pub struct KmeansIndexer<E: node::FloatElement, T: node::IdxType> {
    _dimension: usize,
    _n_center: usize,
    _centers: Vec<Vec<E>>,
    _data_range_begin: usize,
    _data_range_end: usize,
    _mean: T,
    _has_residual: bool,
    _residual: Vec<E>,
    mt: metrics::Metric, //compute metrics
}

impl<E: node::FloatElement, T: node::IdxType> KmeansIndexer<E, T> {
    pub fn new(dimension: usize, n_center: usize, mt: metrics::Metric) -> KmeansIndexer<E, T> {
        KmeansIndexer {
            _dimension: dimension,
            _n_center: n_center,
            _data_range_begin: 0,
            _data_range_end: dimension,
            mt,
            ..Default::default()
        }
    }

    pub fn get_distance_from_vec(&self, x: &node::Node<E, T>, y: &Vec<E>) -> E {
        return metric(
            &x.vectors()[self._data_range_begin..self._data_range_end],
            y,
            self.mt,
        )
        .unwrap();
    }

    pub fn set_residual(&mut self, residual: Vec<E>) {
        self._has_residual = true;
        self._residual = residual;
    }

    pub fn init_center(&mut self, batch_size: usize, batch_data: &Vec<Box<node::Node<E, T>>>) {
        let dimension = self._dimension;
        let n_center = self._n_center;
        let begin = self._data_range_begin;
        let mut mean_center: Vec<E> = Vec::new();
        mean_center.resize(dimension, E::from_f32(0.0).unwrap());

        for i in 0..batch_size {
            let cur_data = batch_data[i].vectors();
            for j in 0..dimension {
                if self._has_residual {
                    mean_center[j] += cur_data[begin + j] - self._residual[begin + j];
                } else {
                    mean_center[j] += cur_data[begin + j];
                }
            }
        }

        for i in 0..dimension {
            mean_center[i] /= E::from_usize(batch_size).unwrap();
        }

        let mut new_centers: Vec<Vec<E>> = Vec::new();
        for i in 0..n_center {
            let mut cur_center: Vec<E> = Vec::new();
            for j in 0..dimension {
                let mut val = mean_center[j];
                if i & (1 << j) == 1 {
                    val += E::from_f32(1.0).unwrap();
                } else {
                    val -= E::from_f32(1.0).unwrap();
                }
                cur_center.push(val);
            }
            new_centers.push(cur_center);
        }
        self._centers = new_centers;
    }

    fn update_center(
        &mut self,
        batch_size: usize,
        batch_data: &Vec<Box<node::Node<E, T>>>,
        assigned_center: &Vec<usize>,
    ) -> Result<Vec<usize>, &'static str> {
        let dimension = self._dimension;
        let n_center = self._n_center;
        let begin = self._data_range_begin;
        let mut new_centers: Vec<Vec<E>> = Vec::new();
        for _i in 0..n_center {
            let mut cur_center: Vec<E> = Vec::new();
            for _j in 0..dimension {
                cur_center.push(E::from_f32(0.0).unwrap());
            }
            new_centers.push(cur_center);
        }
        let mut n_assigned_per_center: Vec<usize> = Vec::new();
        for _i in 0..n_center {
            n_assigned_per_center.push(0);
        }
        for i in 0..batch_size {
            let cur_data = batch_data[i].vectors();
            let cur_center = assigned_center[i];
            n_assigned_per_center[cur_center] += 1;
            for j in 0..dimension {
                if self._has_residual {
                    new_centers[cur_center][j] += cur_data[begin + j] - self._residual[begin + j];
                } else {
                    new_centers[cur_center][j] += cur_data[begin + j];
                }
            }
        }

        for i in 0..n_center {
            if n_assigned_per_center[i] == 0 {
                continue;
            }
            for j in 0..dimension {
                new_centers[i][j] /= E::from_usize(n_assigned_per_center[i]).unwrap();
            }
        }
        self._centers = new_centers;
        Ok(n_assigned_per_center)
    }

    fn search_data(
        &mut self,
        batch_size: usize,
        batch_data: &Vec<Box<node::Node<E, T>>>,
        assigned_center: &mut Vec<usize>,
    ) {
        let n_center = self._n_center;
        let _dimension = self._dimension;
        for i in 0..batch_size {
            let mut nearist_center_id: usize = 0;
            for j in 1..n_center {
                let cur_center = &self._centers[j];
                let nearist_center = &self._centers[nearist_center_id];
                if self.get_distance_from_vec(&batch_data[i], cur_center)
                    < self.get_distance_from_vec(&batch_data[i], nearist_center)
                {
                    nearist_center_id = j;
                }
            }
            assigned_center.push(nearist_center_id);
        }
    }

    fn split_center(
        &mut self,
        batch_size: usize,
        n_assigned_per_center: &mut Vec<usize>,
    ) -> Result<(), &'static str> {
        let dimension = self._dimension;
        let n_center = self._n_center;

        if batch_size == 0 {
            return Err("None to assigned impossible split center");
        }

        for i in 0..n_center {
            if n_assigned_per_center[i] == 0 {
                //rand pick split center
                let mut split_center_id = (i + 1) % n_center;
                loop {
                    let mut rng = rand::thread_rng();
                    let pick_percent =
                        n_assigned_per_center[split_center_id] as f64 / batch_size as f64;
                    if rng.gen_range(0.0, 1.0) < pick_percent {
                        break;
                    }
                    split_center_id = (split_center_id + 1) % n_center;
                }
                const EPS: f32 = 1.0 / 1024.0;
                for j in 0..dimension {
                    if j % 2 == 0 {
                        self._centers[i][j] =
                            self._centers[split_center_id][j] * E::from_f32(1.0 - EPS).unwrap();
                        self._centers[split_center_id][j] *= E::from_f32(1.0 + EPS).unwrap();
                    } else {
                        self._centers[i][j] =
                            self._centers[split_center_id][j] * E::from_f32(1.0 + EPS).unwrap();
                        self._centers[split_center_id][j] *= E::from_f32(1.0 - EPS).unwrap();
                    }
                }
                n_assigned_per_center[i] = n_assigned_per_center[split_center_id] / 2;
                n_assigned_per_center[split_center_id] -= n_assigned_per_center[i];
            }
        }
        Ok(())
    }

    pub fn train(
        &mut self,
        batch_size: usize,
        batch_data: &Vec<Box<node::Node<E, T>>>,
        n_epoch: usize,
    ) {
        self.init_center(batch_size, batch_data);
        for epoch in 0..n_epoch {
            let mut assigned_center: Vec<usize> = Vec::new();
            self.search_data(batch_size, batch_data, &mut assigned_center);
            let mut n_assigned_per_center = self
                .update_center(batch_size, batch_data, &assigned_center)
                .unwrap();
            if epoch < n_epoch - 1 {
                self.split_center(batch_size, &mut n_assigned_per_center)
                    .unwrap();
            }
        }
    }

    pub fn set_range(&mut self, begin: usize, end: usize) {
        assert!(end - begin == self._dimension);
        self._data_range_begin = begin;
        self._data_range_end = end;
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct PQIndex<E: node::FloatElement, T: node::IdxType> {
    _dimension: usize,     //dimension of data
    _n_sub: usize,         //num of subdata
    _sub_dimension: usize, //dimension of subdata
    _sub_bits: usize,      // size of subdata code
    _sub_bytes: usize,     //code save as byte: (_sub_bit + 7)//8
    _n_sub_center: usize,  //num of centers per subdata code
    //n_center_per_sub = 1 << sub_bits
    _code_bytes: usize,         // byte of code
    _train_epoch: usize,        // training epoch
    _centers: Vec<Vec<Vec<E>>>, // size to be _n_sub * _n_sub_center * _sub_dimension
    _is_trained: bool,
    _has_residual: bool,
    _residual: Vec<E>,

    _n_items: usize,
    _max_item: usize,
    _nodes: Vec<Box<node::Node<E, T>>>,
    _assigned_center: Vec<Vec<usize>>,
    mt: metrics::Metric, //compute metrics
    // _item2id: HashMap<i32, usize>,
    _nodes_tmp: Vec<node::Node<E, T>>,
}

impl<E: node::FloatElement, T: node::IdxType> PQIndex<E, T> {
    pub fn new(
        dimension: usize,
        n_sub: usize,
        sub_bits: usize,
        train_epoch: usize,
    ) -> PQIndex<E, T> {
        assert_eq!(dimension % n_sub, 0);
        let sub_dimension = dimension / n_sub;
        let sub_bytes = (sub_bits + 7) / 8;
        assert_eq!(sub_bits <= 32, true);
        let n_center_per_sub = (1 << sub_bits) as usize;
        let code_bytes = sub_bytes * n_sub;
        PQIndex {
            _dimension: dimension,
            _n_sub: n_sub,
            _sub_dimension: sub_dimension,
            _sub_bits: sub_bits,
            _sub_bytes: sub_bytes,
            _n_sub_center: n_center_per_sub,
            _code_bytes: code_bytes,
            _train_epoch: train_epoch,
            _is_trained: false,
            _n_items: 0,
            _max_item: 100000,
            _has_residual: false,
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

    pub fn add_item(&mut self, data: &node::Node<E, T>) -> Result<usize, &'static str> {
        if data.len() != self._dimension {
            return Err("dimension is different");
        }
        // if self._item2id.contains_key(&item) {
        //     //to_do update point
        //     return Ok(self._item2id[&item]);
        // }

        if self._n_items > self._max_item {
            return Err("The number of elements exceeds the specified limit");
        }

        let insert_id = self.init_item(data);
        Ok(insert_id)
    }

    pub fn set_residual(&mut self, residual: Vec<E>) {
        self._has_residual = true;
        self._residual = residual;
    }

    pub fn train_center(&mut self) {
        let n_item = self._n_items;
        let n_sub = self._n_sub;
        for i in 0..n_sub {
            let dimension = self._sub_dimension;
            let n_center = self._n_sub_center;
            let n_epoch = self._train_epoch;
            let begin = i * dimension;
            let end = (i + 1) * dimension;
            let mut cluster = KmeansIndexer::new(dimension, n_center, self.mt);
            cluster.set_range(begin, end);
            if self._has_residual {
                cluster.set_residual(self._residual.to_vec());
            }
            cluster.train(n_item, &self._nodes, n_epoch);
            let mut assigned_center: Vec<usize> = Vec::new();
            cluster.search_data(n_item, &self._nodes, &mut assigned_center);
            self._centers.push(cluster._centers);
            self._assigned_center.push(assigned_center);
        }
        self._is_trained = true;
    }

    pub fn get_distance_from_vec_range(
        &self,
        x: &node::Node<E, T>,
        y: &Vec<E>,
        begin: usize,
        end: usize,
    ) -> E {
        let mut z = x.vectors()[begin..end].to_vec();
        if self._has_residual {
            (begin..end).map(|i| z[i] -= self._residual[i + begin]);
        }
        return metrics::metric(&z, y, self.mt).unwrap();
    }

    pub fn search_knn_adc(
        &self,
        search_data: &node::Node<E, T>,
        k: usize,
    ) -> Result<BinaryHeap<Neighbor<E, usize>>, &'static str> {
        let mut dis2centers: Vec<Vec<E>> = Vec::new();
        for i in 0..self._n_sub {
            let mut sub_dis: Vec<E> = Vec::new();
            for j in 0..self._n_sub_center {
                sub_dis.push(self.get_distance_from_vec_range(
                    search_data,
                    &self._centers[i][j],
                    i * self._sub_dimension,
                    (i + 1) * self._sub_dimension,
                ));
            }
            dis2centers.push(sub_dis);
        }

        let mut top_candidate: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        for i in 0..self._n_items {
            let mut distance = E::from_f32(0.0).unwrap();
            for j in 0..self._n_sub {
                let center_id = self._assigned_center[j][i];
                distance += dis2centers[j][center_id];
            }
            top_candidate.push(Neighbor::new(i, distance));
        }
        while top_candidate.len() > k {
            top_candidate.pop();
        }

        Ok(top_candidate)
    }
}

impl<E: node::FloatElement, T: node::IdxType> ann_index::ANNIndex<E, T> for PQIndex<E, T> {
    fn construct(&mut self, _mt: metrics::Metric) -> Result<(), &'static str> {
        self.mt = _mt;
        self.train_center();
        Result::Ok(())
    }
    fn add_node(&mut self, item: &node::Node<E, T>) -> Result<(), &'static str> {
        match self.add_item(item) {
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
        let mut ret: BinaryHeap<Neighbor<E, usize>> = self.search_knn_adc(item, k).unwrap();
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
        "PQIndex"
    }
}

impl<E: node::FloatElement + DeserializeOwned, T: node::IdxType + DeserializeOwned>
    ann_index::SerializableIndex<E, T> for PQIndex<E, T>
{
    fn load(path: &str, _args: &arguments::Args) -> Result<Self, &'static str> {
        let file = File::open(path).unwrap_or_else(|_| panic!("unable to open file {:?}", path));
        let mut instance: PQIndex<E, T> = bincode::deserialize_from(&file).unwrap();
        instance._nodes = instance
            ._nodes_tmp
            .iter()
            .map(|x| Box::new(x.clone()))
            .collect();
        Ok(instance)
    }

    fn dump(&mut self, path: &str, _args: &arguments::Args) -> Result<(), &'static str> {
        self._nodes_tmp = self._nodes.iter().map(|x| *x.clone()).collect();
        let encoded_bytes = bincode::serialize(&self).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&encoded_bytes)
            .unwrap_or_else(|_| panic!("unable to write file {:?}", path));
        Result::Ok(())
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct IVFPQIndex<E: node::FloatElement, T: node::IdxType> {
    _dimension: usize,     //dimension of data
    _n_sub: usize,         //num of subdata
    _sub_dimension: usize, //dimension of subdata
    _sub_bits: usize,      // size of subdata code
    _sub_bytes: usize,     //code save as byte: (_sub_bit + 7)//8
    _n_sub_center: usize,  //num of centers per subdata code
    //n_center_per_sub = 1 << sub_bits
    _code_bytes: usize,  // byte of code
    _train_epoch: usize, // training epoch
    _search_n_center: usize,
    _n_kmeans_center: usize,
    _centers: Vec<Vec<E>>,
    _ivflist: Vec<Vec<usize>>, //ivf center id
    _pq_list: Vec<PQIndex<E, T>>,
    _is_trained: bool,

    _n_items: usize,
    _max_item: usize,
    _nodes: Vec<Box<node::Node<E, T>>>,
    _assigned_center: Vec<Vec<usize>>,
    mt: metrics::Metric, //compute metrics
    // _item2id: HashMap<i32, usize>,
    _nodes_tmp: Vec<node::Node<E, T>>,
}

impl<E: node::FloatElement, T: node::IdxType> IVFPQIndex<E, T> {
    pub fn new(
        dimension: usize,
        n_sub: usize,
        sub_bits: usize,
        n_kmeans_center: usize,
        search_n_center: usize,
        train_epoch: usize,
    ) -> IVFPQIndex<E, T> {
        assert_eq!(dimension % n_sub, 0);
        let sub_dimension = dimension / n_sub;
        let sub_bytes = (sub_bits + 7) / 8;
        assert_eq!(sub_bits <= 32, true);
        let n_center_per_sub = (1 << sub_bits) as usize;
        let code_bytes = sub_bytes * n_sub;
        let mut ivflist: Vec<Vec<usize>> = Vec::new();
        for _i in 0..n_kmeans_center {
            let ivf: Vec<usize> = Vec::new();
            ivflist.push(ivf);
        }
        IVFPQIndex {
            _dimension: dimension,
            _n_sub: n_sub,
            _sub_dimension: sub_dimension,
            _sub_bits: sub_bits,
            _sub_bytes: sub_bytes,
            _n_sub_center: n_center_per_sub,
            _code_bytes: code_bytes,
            _n_kmeans_center: n_kmeans_center,
            _search_n_center: search_n_center,
            _ivflist: ivflist,
            _train_epoch: train_epoch,
            _is_trained: false,
            _n_items: 0,
            _max_item: 100000,
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

    pub fn add_item(&mut self, data: &node::Node<E, T>) -> Result<usize, &'static str> {
        if data.len() != self._dimension {
            return Err("dimension is different");
        }
        // if self._item2id.contains_key(&item) {
        //     //to_do update point
        //     return Ok(self._item2id[&item]);
        // }

        if self._n_items > self._max_item {
            return Err("The number of elements exceeds the specified limit");
        }

        let insert_id = self.init_item(data);
        Ok(insert_id)
    }

    pub fn train(&mut self) {
        let n_item = self._n_items;
        let dimension = self._dimension;
        let n_center = self._n_kmeans_center;
        let n_epoch = self._train_epoch;
        let mut cluster = KmeansIndexer::new(dimension, n_center, self.mt);
        cluster.set_range(0, dimension);
        cluster.train(n_item, &self._nodes, n_epoch);
        let mut assigned_center: Vec<usize> = Vec::new();
        cluster.search_data(n_item, &self._nodes, &mut assigned_center);
        self._centers = cluster._centers;
        for i in 0..n_item {
            let center_id = assigned_center[i];
            self._ivflist[center_id].push(i);
        }

        for i in 0..n_center {
            let mut center_pq = PQIndex::<E, T>::new(
                self._dimension,
                self._n_sub,
                self._sub_bits,
                self._train_epoch,
            );

            for j in 0..self._ivflist[i].len() {
                center_pq.add_item(&self._nodes[self._ivflist[i][j]].clone());
            }

            center_pq.set_residual(self._centers[i].to_vec());
            center_pq.train_center();
            self._pq_list.push(center_pq);
        }

        self._is_trained = true;
    }

    pub fn get_distance_from_vec_range(
        &self,
        x: &node::Node<E, T>,
        y: &Vec<E>,
        begin: usize,
        end: usize,
    ) -> E {
        return metrics::metric(&x.vectors()[begin..end], y, self.mt).unwrap();
    }

    pub fn search_knn_adc(
        &self,
        search_data: &node::Node<E, T>,
        k: usize,
    ) -> Result<BinaryHeap<Neighbor<E, usize>>, &'static str> {
        let mut top_centers: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        let n_kmeans_center = self._n_kmeans_center;
        let dimension = self._dimension;
        for i in 0..n_kmeans_center {
            top_centers.push(Neighbor::new(
                i,
                -self.get_distance_from_vec_range(search_data, &self._centers[i], 0, dimension),
            ))
        }

        let mut top_candidate: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        for _i in 0..self._search_n_center {
            let center = top_centers.pop().unwrap().idx();
            let mut ret = self._pq_list[center]
                .search_knn_adc(search_data, k)
                .unwrap();
            while !ret.is_empty() {
                let ret_peek = ret.pop().unwrap();
                top_candidate.push(ret_peek);
                if top_candidate.len() > k {
                    top_candidate.pop();
                }
            }
        }

        Ok(top_candidate)
    }
}
