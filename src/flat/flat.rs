use node::Node;

use crate::annoy::random;
use crate::common::metrics;
use crate::common::neighbor;
use crate::common::node;
use std::collections::BinaryHeap;
pub struct FlatIndex<E: node::FloatElement> {
    nodes: Vec<Box<node::Node<E>>>,
}

impl<E: node::FloatElement> FlatIndex<E> {
    fn new() -> FlatIndex<E> {
        FlatIndex::<E> { nodes: Vec::new() }
    }
    pub fn train(&self) {}

    pub fn add(&mut self, item: &node::Node<E>) {
        self.nodes.push(Box::new(item.clone()));
    }

    pub fn search_k_node<F>(&self, item: &node::Node<E>, k: usize, metrics: &F) -> Vec<(Node<E>, E)>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>,
    {
        let mut heap = BinaryHeap::new();
        for i in 0..self.nodes.len() {
            heap.push(neighbor::Neighbor::new(
                i,
                item.distance(&self.nodes[i], metrics).unwrap(),
            ));
            if heap.len() > k {
                heap.pop();
            }
        }

        let mut result = Vec::new();
        for neighbor in heap.iter() {
            result.push((*self.nodes[neighbor.idx()].clone(), neighbor.distance()))
        }
        result
    }

    pub fn search_k<F>(&self, is: &[E], k: usize, metrics: &F) -> Vec<(Vec<E>, E)>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>,
    {
        let n = Node::new(is);
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
        for neighbor in heap.iter() {
            result.push((
                self.nodes[neighbor.idx()].vectors().clone(),
                neighbor.distance(),
            ))
        }
        result
    }
}
