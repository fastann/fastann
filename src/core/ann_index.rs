use crate::core::metrics;
use crate::core::node;

pub trait AnnIndex<E: node::FloatElement> {
    fn construct(&self); // construct algorithm structure
    fn add(&mut self, item: &node::Node<E>);
    fn once_constructed(&self) -> bool; // has already been constructed?
    fn reconstruct(&mut self);
    fn search_node<F>(
        &self,
        item: &node::Node<E>,
        k: usize,
        metrics: &F,
    ) -> Vec<(node::Node<E>, E)>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>;

    fn search<F>(&self, item: &[E], k: usize, metrics: &F) -> Vec<(node::Node<E>, E)>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>;

    fn load(&self, path: &str) -> Result<(), &'static str>;

    fn dump(&self, path: &str) -> Result<(), &'static str>;
}
