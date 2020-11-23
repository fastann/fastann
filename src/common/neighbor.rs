extern crate num;
use std::cmp::Ordering;
#[derive(Default, Clone, PartialEq, Debug)]
pub struct Neighbor {
    pub _idx: usize,
    pub _distance: f64,
}

impl Neighbor {
    pub fn new(idx: usize, distance: f64) -> Neighbor {
        return Neighbor {
            _idx: idx,
            _distance: distance,
        };
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Neighbor) -> Ordering {
        let ord = if self._distance > other._distance {
            Ordering::Greater
        } else if self._distance < other._distance {
            Ordering::Less
        } else {
            Ordering::Equal
        };
        return ord;
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Neighbor) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Neighbor {}
