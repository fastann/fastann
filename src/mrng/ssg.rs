use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use core::cmp::Reverse;

extern crate num;
use core::cmp::Ordering;

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
    fn partial_cmp(&self, other: &Neighbor<E, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E: node::FloatElement, T: node::IdxType> Eq for SubNeighbor<E, T> {}

pub struct SatelliteSystemGraphIndex<E: node::FloatElement, T: node::IdxType> {
    nodes: Vec<Box<node::Node<E, T>>>,
    mt: metrics::Metric,
    dimension: usize,
    L:usize,
    final_graph_ :Vec<Vec<usize>>,
}

impl<E: node::FloatElement, T: node::IdxType> SatelliteSystemGraphIndex<E, T> {
    pub fn new(dimension: usize, L: usize) -> SatelliteSystemGraphIndex<E, T> {
        SatelliteSystemGraphIndex::<E, T> {
            nodes: Vec::new(),
            mt: metrics::Metric::Unknown,
            dimension: dimension,
            L: L,
        }
    }

    fn initialize_graph(&mut self) {
        let mut center = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            center.push(0.);
        }
        for i in 0..self.nodes.len() {
            for j in 0..self.dimension {
                center[j] += self.nodes[i].vectors()[j];
            }
        }
        for j in 0..self.dimension {
            center[j] /= self.nodes.len() as E;
        }
    }

    fn get_random_nodes_idx(&self, indices: &[usize]) {
        let mut rng = rand::thread_rng();
        for i in 0..indices.len() {
            indices[i] = rng.gen_range(0, self.nodes.len() - indices.len());
        }
        indices.sort();
        for i in 1..indices.len() {
            if indices[i] <= indices[i-1] {
                indices[i] = indices[i-1] + 1;
            }
        }
        usize offset = rng.gen_range(0, self.nodes.len());
        for i in 0..indices.len() {
            indices[i] = (indices + offset) % self.nodes.len();
        }
    }

    fn get_point_neighbors(&self, item: &node::Node<E, T>) {
        let mut return_set = Vec::new();
        let mut init_ids = [0;self.L];
        self.get_random_nodes_idx(&init_ids);
        let mut flags = [0;self.nodes.len()]; // as bitmap;
        let _L = 0;
        for id in init_ids.iter() {
            if id > self.nodes.len() {
                continue;
            }
            return_set.push((SubNeighbor::new(
                id,
                item.metric(&self.nodes[id], self.mt).unwrap(),
                true
            ));
            flags[id] = 1;
            _L +=1;
        }
        return_set.sort_by(|x| x.distance);
        let mut k =0;
        while k < _L {
            let nk = _L;
            if retset[k].flag {
                return_set[k].flag = false;
                let n = retset[k].id,

                for m in 0..final_graph_
            }
        }
    }
}

impl<E: node::FloatElement, T: node::IdxType> ann_index::ANNIndex<E, T> for SatelliteSystemGraphIndex<E, T> {
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
