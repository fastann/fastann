use crate::core::ann_index;
use crate::core::arguments;
use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use core::cmp::Reverse;

pub struct BruteForceIndex<E: node::FloatElement, T: node::IdxType> {
    nodes: Vec<Box<node::Node<E, T>>>,
    mt: metrics::Metric,
}

impl<E: node::FloatElement, T: node::IdxType> BruteForceIndex<E, T> {
    pub fn new() -> BruteForceIndex<E, T> {
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
        // let since_the_epoch = SystemTime::now()
        //     .duration_since(start)
        //     .expect("Time went backwards");
        // println!("{:?}: {:?}", "general", since_the_epoch);

        // let start = SystemTime::now();
        // let atomic_heap = Arc::new(Mutex::new(BinaryHeap::new()));
        // (0..self.nodes.len()).into_par_iter().for_each(|i| {
        //     let m = item.metric(&self.nodes[i], self.mt).unwrap();
        //     atomic_heap.lock().unwrap().push(neighbor::Neighbor::new(
        //         // use max heap, and every time pop out the greatest one in the heap
        //         i, m,
        //     ));
        //     if atomic_heap.lock().unwrap().len() > k {
        //         atomic_heap.lock().unwrap().pop().unwrap();
        //     }
        // });
        // let since_the_epoch = SystemTime::now()
        //     .duration_since(start)
        //     .expect("Time went backwards");
        // println!("{:?}: {:?}", "parallelism", since_the_epoch);

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
