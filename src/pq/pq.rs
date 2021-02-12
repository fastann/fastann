use crate::core::metrics;
use crate::core::neighbor::Neighbor;
use rand::prelude::*;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Default, Clone, PartialEq, Debug)]
pub struct Data {
    _demension: i32,
    _val: Vec<f64>,
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
        let ret = metrics::euclidean_distance(&(self._val), &(data._val));
        ret
    }
}

#[derive(Default, Debug)]
pub struct KmeansIndexer {
    _demension: usize,
    _n_center: usize,
    _n_epoch: usize,
    _centers: Vec<Vec<f64>>,
    _data_range_begin: usize,
    _data_range_end: usize,
}

impl KmeansIndexer {
    pub fn new(demension: usize, n_center: usize, _n_epoch: usize) -> KmeansIndexer {
        return KmeansIndexer {
            _demension: demension,
            _n_center: n_center,
            _n_epoch: _n_epoch,
            _data_range_begin: 0,
            _data_range_end: demension,
            ..Default::default()
        };
    }

    pub fn get_distance_from_vec(&self, x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        return metrics::euclidean_distance_range(
            x,
            y,
            self._data_range_begin,
            self._data_range_end,
        )
        .unwrap();
    }

    pub fn init_center(&mut self, batch_size: usize, batch_data: &Vec<Vec<f64>>) {
        let demension = self._demension;
        let n_center = self._n_center;
        let begin = self._data_range_begin;
        let mut mean_center: Vec<f64> = Vec::new();
        for i in 0..demension {
            mean_center.push(0.0);
        }

        for i in 0..batch_size {
            let cur_data = &batch_data[i];
            for j in 0..demension {
                mean_center[j] += cur_data[begin + j];
            }
        }

        for i in 0..demension {
            mean_center[i] /= batch_size as f64;
        }

        let mut new_centers: Vec<Vec<f64>> = Vec::new();
        for i in 0..n_center {
            let mut cur_center: Vec<f64> = Vec::new();
            for j in 0..demension {
                let mut val = mean_center[j];
                if i & (1 << j) == 1 {
                    val += 1.0;
                } else {
                    val -= 1.0;
                }
                cur_center.push(val);
            }
            new_centers.push(cur_center);
        }
    }

    fn update_center(
        &mut self,
        batch_size: usize,
        batch_data: &Vec<Vec<f64>>,
        assigned_center: &Vec<usize>,
    ) -> Result<Vec<usize>, &'static str> {
        let demension = self._demension;
        let n_center = self._n_center;
        let begin = self._data_range_begin;
        let mut new_centers: Vec<Vec<f64>> = Vec::new();
        for i in 0..n_center {
            let mut cur_center: Vec<f64> = Vec::new();
            for j in 0..demension {
                cur_center.push(0.0);
            }
            new_centers.push(cur_center);
        }
        let mut n_assigned_per_center: Vec<usize> = Vec::new();
        for i in 0..n_center {
            n_assigned_per_center.push(0);
        }
        for i in 0..batch_size {
            let cur_data = &batch_data[i];
            let cur_center = assigned_center[i];
            n_assigned_per_center[cur_center] += 1;
            for j in 0..demension {
                new_centers[cur_center][j] += cur_data[begin + j];
            }
        }

        for i in 0..n_center {
            if n_assigned_per_center[i] == 0 {
                continue;
            }
            for j in 0..demension {
                new_centers[i][j] /= n_assigned_per_center[i] as f64;
            }
        }
        self._centers = new_centers;
        return Ok(n_assigned_per_center);
    }

    fn search_data(
        &mut self,
        batch_size: usize,
        batch_data: &Vec<Vec<f64>>,
        assigned_center: &mut Vec<usize>,
    ) {
        let n_center = self._n_center;
        let demension = self._demension;
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
            assigned_center[i] = nearist_center_id;
        }
    }

    fn split_center(
        &mut self,
        batch_size: usize,
        n_assigned_per_center: &mut Vec<usize>,
    ) -> Result<(), &'static str> {
        let demension = self._demension;
        let n_center = self._n_center;

        if batch_size == 0 {
            return Err("None to assigned impossible split center");
        }

        for i in 0..n_center {
            if n_assigned_per_center[i] == 0 {
                //rand pick split center
                let mut split_center_id = i + 1;
                loop {
                    let mut rng = rand::thread_rng();
                    let pick_percent =
                        n_assigned_per_center[split_center_id] as f64 / batch_size as f64;
                    if rng.gen_range(0.0, 1.0) < pick_percent {
                        break;
                    }
                    split_center_id = (split_center_id + 1) % n_center;
                }
                let EPS = 1.0 / 1024.0;
                for j in 0..demension {
                    if j % 2 == 0 {
                        self._centers[i][j] = self._centers[split_center_id][j] * (1.0 - EPS);
                        self._centers[split_center_id][j] =
                            self._centers[split_center_id][j] * (1.0 + EPS);
                    } else {
                        self._centers[i][j] = self._centers[split_center_id][j] * (1.0 + EPS);
                        self._centers[split_center_id][j] =
                            self._centers[split_center_id][j] * (1.0 - EPS);
                    }
                }
                n_assigned_per_center[i] = n_assigned_per_center[split_center_id] / 2;
                n_assigned_per_center[split_center_id] -= n_assigned_per_center[i];
            }
        }
        return Ok(());
    }

    pub fn train(&mut self, batch_size: usize, batch_data: &Vec<Vec<f64>>) {
        self.init_center(batch_size, batch_data);
        for epoch in 0..self._n_epoch {
            let mut assigned_center: Vec<usize> = Vec::new();
            self.search_data(batch_size, batch_data, &mut assigned_center);
            let mut n_assigned_per_center = self
                .update_center(batch_size, batch_data, &assigned_center)
                .unwrap();
            if epoch < self._n_epoch - 1 {
                self.split_center(batch_size, &mut n_assigned_per_center);
            }
        }
    }

    pub fn set_range(&mut self, begin: usize, end: usize) {
        assert!(begin >= 0 && end - begin == self._demension);
        self._data_range_begin = begin;
        self._data_range_end = end;
    }
}

#[derive(Default, Debug)]
pub struct PQIndexer {
    _demension: usize,     //dimension of data
    _n_sub: usize,         //num of subdata
    _sub_demension: usize, //dimension of subdata
    _sub_bits: usize,      // size of subdata code
    _sub_bytes: usize,     //code save as byte: (_sub_bit + 7)//8
    _n_sub_center: usize,  //num of centers per subdata code
    //n_center_per_sub = 1 << sub_bits
    _codebytes: usize,            // byte of code
    _train_epoch: usize,          // training epoch
    _centers: Vec<Vec<Vec<f64>>>, // size to be _n_sub * _n_sub_center * _sub_demension
    _is_trained: bool,

    _n_items: usize,
    _max_item: usize,
    _datas: Vec<Vec<f64>>,
    _assigned_center: Vec<Vec<usize>>,
    _item2id: HashMap<i32, usize>,
}

impl PQIndexer {
    pub fn new(demension: usize, n_sub: usize, sub_bits: usize, train_epoch: usize) -> PQIndexer {
        assert_eq!(demension % n_sub, 0);
        let sub_demension = demension / n_sub;
        let sub_bytes = (sub_bits + 7) / 8;
        assert_eq!(sub_bits <= 32, true);
        let n_center_per_sub = (1 << sub_bits) as usize;
        let codebytes = sub_bytes * n_sub;
        return PQIndexer {
            _demension: demension,
            _n_sub: n_sub,
            _sub_demension: sub_demension,
            _sub_bits: sub_bits,
            _sub_bytes: sub_bytes,
            _n_sub_center: n_center_per_sub,
            _codebytes: codebytes,
            _train_epoch: train_epoch,
            _is_trained: false,
            _n_items: 0,
            _max_item: 100000,
            ..Default::default()
        };
    }

    pub fn init_item(&mut self, item: i32, data: &[f64]) -> usize {
        let cur_id = self._n_items;
        self._item2id.insert(item, cur_id);
        self._datas.push(data.to_vec());
        self._n_items += 1;
        return cur_id;
    }

    pub fn add_item(&mut self, item: i32, data: &[f64]) -> Result<usize, &'static str> {
        if data.len() != self._demension {
            return Err("dimension is different");
        }
        if self._item2id.contains_key(&item) {
            //to_do update point
            return Ok(self._item2id[&item]);
        }

        if self._n_items > self._max_item {
            return Err("The number of elements exceeds the specified limit");
        }

        let insert_id = self.init_item(item, data);
        return Ok(insert_id);
    }

    pub fn train_center(&mut self) {
        let n_item = self._n_items;
        let n_sub = self._n_sub;
        for i in 0..n_sub {
            let demension = self._sub_demension;
            let n_center = self._n_sub_center;
            let n_epoch = self._train_epoch;
            let begin = n_sub * demension;
            let end = (n_sub + 1) * demension;
            let mut clus = KmeansIndexer::new(demension, n_center, n_epoch);
            clus.set_range(begin, end);
            clus.train(n_item, &self._datas);
            let mut assigned_center: Vec<usize> = Vec::new();
            clus.search_data(n_item, &self._datas, &mut assigned_center);

            self._centers.push(clus._centers);
            self._assigned_center.push(assigned_center);
        }
        self._is_trained = true;
    }

    pub fn get_distance_from_vec_range(
        &self,
        x: &Vec<f64>,
        y: &Vec<f64>,
        begin: usize,
        end: usize,
    ) -> f64 {
        return metrics::euclidean_distance_range(x, y, begin, end).unwrap();
    }

    pub fn search_knn_ADC(
        &self,
        search_data: &Vec<f64>,
        k: usize,
    ) -> Result<BinaryHeap<Neighbor<f64, usize>>, &'static str> {
        let mut dis2centers: Vec<Vec<f64>> = Vec::new();
        for i in 0..self._n_sub {
            let mut sub_dis: Vec<f64> = Vec::new();
            for j in 0..self._n_sub_center {
                sub_dis.push(self.get_distance_from_vec_range(
                    search_data,
                    &self._centers[i][j],
                    i * self._n_sub,
                    (i + 1) * self._n_sub,
                ));
            }
            dis2centers.push(sub_dis);
        }

        let mut top_candidate: BinaryHeap<Neighbor<f64, usize>> = BinaryHeap::new();
        for i in 0..self._n_items {
            let mut distance = 0.0;
            for j in 0..self._n_sub {
                let center_id = self._assigned_center[j][i];
                distance += dis2centers[j][center_id];
            }
            top_candidate.push(Neighbor::new(i, -distance));
        }
        while top_candidate.len() > k {
            top_candidate.pop();
        }

        return Ok(top_candidate);
    }
}
