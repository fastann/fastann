use crate::annoy::random;
use crate::core::ann_index;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use crate::core::parameters;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

pub struct FlatIndex<E: node::FloatElement> {
    nodes: Vec<Box<node::Node<E>>>,
}

impl<E: node::FloatElement> FlatIndex<E> {
    pub fn new(p: parameters::Parameters) -> FlatIndex<E> {
        FlatIndex::<E> { nodes: Vec::new() }
    }
}

impl<E: node::FloatElement> ann_index::AnnIndex<E> for FlatIndex<E> {
    fn construct(&self) {}
    fn add(&mut self, item: &node::Node<E>) {
        self.nodes.push(Box::new(item.clone()));
    }
    fn once_constructed(&self) -> bool {
        true
    }
    fn reconstruct(&mut self) {}
    fn search_node<F>(&self, item: &node::Node<E>, k: usize, metrics: &F) -> Vec<(node::Node<E>, E)>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>,
    {
        let mut heap = BinaryHeap::new();
        for i in 0..self.nodes.len() {
            heap.push(neighbor::Neighbor::new( // use max heap, and every time pop out the greatest one in the heap
                i,
                item.distance(&self.nodes[i], metrics).unwrap(),
            ));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut result = Vec::new();
        for neighbor_rev in heap.iter().rev() {
            result.push((*self.nodes[neighbor_rev.idx()].clone(), neighbor_rev.distance()))
        }
        result
    }

    fn search<F>(&self, item: &[E], k: usize, metrics: &F) -> Vec<(node::Node<E>, E)>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>,
    {
        let n = node::Node::new(item);
        let mut heap = BinaryHeap::new();
        for i in 0..self.nodes.len() {
            heap.push(neighbor::Neighbor::new(
                i,
                n.distance(&self.nodes[i], metrics).unwrap(),
            ));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut result = Vec::new();
        for neighbor_rev in heap.iter().rev() {
            result.push((*self.nodes[neighbor_rev.idx()].clone(), neighbor_rev.distance()))
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
