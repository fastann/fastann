use crate::core::ann_index;
use crate::core::arguments;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use crate::core::parameters;
use core::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct BruteForceIndex<E: node::FloatElement, T: node::IdxType> {
    nodes: Vec<Box<node::Node<E, T>>>,
    mt: metrics::Metric,
}

impl<E: node::FloatElement, T: node::IdxType> BruteForceIndex<E, T> {
    pub fn new(p: parameters::Parameters) -> BruteForceIndex<E, T> {
        BruteForceIndex::<E, T> {
            nodes: Vec::new(),
            mt: metrics::Metric::Unknown,
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
        args: &arguments::Arguments,
    ) -> Vec<(node::Node<E, T>, E)> {
        let mut heap = BinaryHeap::new();
        let mut base = E::default();
        for i in 0..self.nodes.len() {
            heap.push(neighbor::Neighbor::new(
                // use max heap, and every time pop out the greatest one in the heap
                i,
                item.metric(&self.nodes[i], self.mt).unwrap(),
            ));
            if heap.len() > k {
                let xp = heap.pop().unwrap();
                if xp.distance() > base {
                    base = xp.distance();
                }
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
