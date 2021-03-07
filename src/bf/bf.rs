use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;

#[derive(Debug, Serialize, Deserialize)]
pub struct BruteForceIndex<E: node::FloatElement, T: node::IdxType> {
    #[serde(skip_serializing, skip_deserializing)]
    nodes: Vec<Box<node::Node<E, T>>>,
    tmp_nodes: Vec<node::Node<E, T>>, // only use for serialization scene
    mt: metrics::Metric,
}

impl<E: node::FloatElement, T: node::IdxType> BruteForceIndex<E, T> {
    pub fn new() -> BruteForceIndex<E, T> {
        BruteForceIndex::<E, T> {
            nodes: Vec::new(),
            mt: metrics::Metric::Unknown,
            tmp_nodes: Vec::new(),
        }
    }
}

impl<E: node::FloatElement, T: node::IdxType> ann_index::ANNIndex<E, T> for BruteForceIndex<E, T> {
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
        // let start = SystemTime::now();
        let mut heap = BinaryHeap::new();
        for i in 0..self.nodes.len() {
            heap.push(neighbor::Neighbor::new(
                // use max heap, and every time pop out the greatest one in the heap
                i,
                item.metric(&self.nodes[i], self.mt).unwrap(),
            ));
            if heap.len() > k {
                let xp = heap.pop().unwrap();
            }
        }

        let mut result = Vec::new();
        while !heap.is_empty() {
            let neighbor_rev = heap.pop().unwrap();
            result.push((
                *self.nodes[neighbor_rev.idx()].clone(),
                neighbor_rev.distance(),
            ))
        }
        result.reverse();
        result
    }

    fn name(&self) -> &'static str {
        "BruteForceIndex"
    }
}

impl<E: node::FloatElement + DeserializeOwned, T: node::IdxType + DeserializeOwned>
    ann_index::SerializableANNIndex<E, T> for BruteForceIndex<E, T>
{
    fn load(path: &str, args: &arguments::Args) -> Result<Self, &'static str> {
        let mut file = File::open(path).expect(&format!("unable to open file {:?}", path));
        let mut instance: BruteForceIndex<E, T> = bincode::deserialize_from(file).unwrap();
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
