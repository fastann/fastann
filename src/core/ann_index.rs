use crate::core::arguments;
use crate::core::metrics;
use crate::core::node;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

pub trait ANNIndex<E: node::FloatElement, T: node::IdxType>: Send + Sync {
    fn construct(&mut self, mt: metrics::Metric) -> Result<(), &'static str>; // construct algorithm structure
    fn add_node(&mut self, item: &node::Node<E, T>) -> Result<(), &'static str>;
    fn add_without_idx(&mut self, vs: &[E]) -> Result<(), &'static str> {
        let n = node::Node::new(vs);
        self.add_node(&n)
    }
    fn add(&mut self, vs: &[E], idx: T) -> Result<(), &'static str> {
        let n = node::Node::new_with_idx(vs, idx);
        self.add_node(&n)
    }

    fn batch_add(&mut self, vss: &[&[E]], indices: &[T]) -> Result<(), &'static str> {
        for idx in 0..vss.len() {
            let n = node::Node::new_with_idx(vss[idx], indices[idx].clone());
            match self.add_node(&n) {
                Err(err) => return Err(err),
                _ => (),
            }
        }
        Ok(())
    }
    fn once_constructed(&self) -> bool; // has already been constructed?
    fn reconstruct(&mut self, mt: metrics::Metric);
    fn node_search_k(
        &self,
        item: &node::Node<E, T>,
        k: usize,
        args: &arguments::Args,
    ) -> Vec<(node::Node<E, T>, E)>;

    // e.g.
    //idx.node_search_k(
    //     &n,
    //     k,
    //     &arguments::Args::new()
    //         .fset("hello", 0.1)
    //         .iset("word", 2)
    //         .fset("aljun", 0.2)
    //         .sset("goodbye", "my_key"),
    // )
    fn search_k(&self, item: &[E], k: usize) -> Vec<(node::Node<E, T>, E)> {
        let n = node::Node::new(item);
        self.node_search_k(&n, k, &arguments::Args::new())
    }

    fn search_k_with_args(
        &self,
        item: &[E],
        k: usize,
        args: &arguments::Args,
    ) -> Vec<(node::Node<E, T>, E)> {
        let n = node::Node::new(item);
        self.node_search_k(&n, k, args)
    }

    fn load(&mut self, path: &str) -> Result<(), &'static str>;

    fn dump(&self, path: &str) -> Result<(), &'static str>;

    fn name(&self) -> &'static str;
}

pub trait SerializableANNIndex<E: node::FloatElement + DeserializeOwned, T: node::IdxType + DeserializeOwned>: Send + Sync + ANNIndex<E,T> {

    fn load(path: &str) -> Result<Self, &'static str> where Self: Sized {
        Err("empty")
    }
}

// fn load_ann_idx<
//     'a,
//     E: node::FloatElement + Deserialize<'a>,
//     T: node::IdxType + Deserialize<'a>,
//     A: ANNIndex<E, T>
// >(
//     path: &str,
// ) -> Result<Box<A>, &'static str> {
//     Err("www")
// }
