use crate::core::ann_index;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use crate::core::parameters;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct FlatIndex<E: node::FloatElement, T: node::KeyType> {
    nodes: Vec<Box<node::Node<E, T>>>,
}

impl<E: node::FloatElement, T: node::KeyType> FlatIndex<E, T> {
    pub fn new(p: parameters::Parameters) -> FlatIndex<E, T> {
        FlatIndex::<E, T> { nodes: Vec::new() }
    }
}

impl<E: node::FloatElement, T: node::KeyType> ann_index::AnnIndex<E, T> for FlatIndex<E, T> {
    fn construct(&self) {}
    fn add(&mut self, item: &node::Node<E, T>) {
        self.nodes.push(Box::new(item.clone()));
    }
    fn once_constructed(&self) -> bool {
        true
    }
    fn reconstruct(&mut self) {}
    fn search_node(
        &self,
        item: &node::Node<E, T>,
        k: usize,
        mt: metrics::MetricType,
    ) -> Vec<(node::Node<E, T>, E)> {
        let mut heap = BinaryHeap::new();
        for i in 0..self.nodes.len() {
            heap.push(neighbor::Neighbor::new(
                // use max heap, and every time pop out the greatest one in the heap
                i,
                item.metric(&self.nodes[i], mt).unwrap(),
            ));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut result = Vec::new();
        for neighbor_rev in heap.iter().rev() {
            result.push((
                *self.nodes[neighbor_rev.idx()].clone(),
                neighbor_rev.distance(),
            ))
        }
        result
    }

    fn search(&self, item: &[E], k: usize, mt: metrics::MetricType) -> Vec<(node::Node<E, T>, E)> {
        let n = node::Node::new(item);
        let mut heap = BinaryHeap::new();
        for i in 0..self.nodes.len() {
            heap.push(neighbor::Neighbor::new(
                i,
                n.metric(&self.nodes[i], mt).unwrap(),
            ));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut result = Vec::new();
        for neighbor_rev in heap.iter().rev() {
            result.push((
                *self.nodes[neighbor_rev.idx()].clone(),
                neighbor_rev.distance(),
            ))
        }
        result
    }

    fn load(&self, path: &str) -> Result<(), &'static str> {
        std::result::Result::Ok(())
    }

    fn dump(&self, path: &str) -> Result<(), &'static str> {
        std::result::Result::Ok(())
    }
}
