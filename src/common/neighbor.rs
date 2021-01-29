extern crate num;
use crate::common::node;
use std::cmp::Ordering;
#[derive(Default, Clone, PartialEq, Debug)]
pub struct Neighbor<E: node::Element> {
    pub _idx: usize,
    pub _distance: E,
}

impl<E: node::Element> Neighbor<E> {
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

impl<E: node::Element> Ord for Neighbor<E> {
    fn cmp(&self, other: &Neighbor<E>) -> Ordering {
        let ord = if self._distance > other._distance {
            Ordering::Greater
        } else if self._distance < other._distance {
            Ordering::Less
        } else {
            Ordering::Equal
        };
        panic!("invalid distance {:?}", self._distance);
    }
}

impl<E: node::Element> PartialOrd for Neighbor<E> {
    fn partial_cmp(&self, other: &Neighbor<E>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E: node::Element> Eq for Neighbor<E> {}
