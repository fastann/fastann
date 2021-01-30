extern crate num;
use crate::common::node;
use std::cmp::Ordering;
#[derive(Default, Clone, PartialEq, Debug)]
pub struct Neighbor<E: node::FloatElement> {
    pub _idx: usize,
    pub _distance: E,
}

impl<E: node::FloatElement> Neighbor<E> {
    pub fn new(idx: usize, distance: E) -> Neighbor<E> {
        return Neighbor {
            _idx: idx,
            _distance: distance,
        };
    }

    pub fn idx(&self) -> usize {
        self._idx
    }

    pub fn distance(&self) -> E {
        self._distance
    }
}

impl<E: node::FloatElement> Ord for Neighbor<E> {
    fn cmp(&self, other: &Neighbor<E>) -> Ordering {
        self._distance.partial_cmp(&other._distance).unwrap()
    }
}

impl<E: node::FloatElement> PartialOrd for Neighbor<E> {
    fn partial_cmp(&self, other: &Neighbor<E>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E: node::FloatElement> Eq for Neighbor<E> {}
