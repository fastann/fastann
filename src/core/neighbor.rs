extern crate num;
use crate::core::node;
use std::cmp::Ordering;
#[derive(Default, Clone, PartialEq, Debug)]
pub struct Neighbor<E: node::FloatElement, T: node::KeyType> {
    pub _key: T,
    pub _distance: E,
}

impl<E: node::FloatElement, T: node::KeyType> Neighbor<E, T> {
    pub fn new(key: T, distance: E) -> Neighbor<E, T> {
        return Neighbor {
            _key: key,
            _distance: distance,
        };
    }

    pub fn key(&self) -> T {
        self._key.clone()
    }

    pub fn distance(&self) -> E {
        self._distance
    }
}

impl<E: node::FloatElement, T: node::KeyType> Ord for Neighbor<E, T> {
    fn cmp(&self, other: &Neighbor<E, T>) -> Ordering {
        self._distance.partial_cmp(&other._distance).unwrap()
    }
}

impl<E: node::FloatElement, T: node::KeyType> PartialOrd for Neighbor<E, T> {
    fn partial_cmp(&self, other: &Neighbor<E, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E: node::FloatElement, T: node::KeyType> Eq for Neighbor<E, T> {}
