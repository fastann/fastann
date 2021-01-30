use crate::common::node;
use crate::common::metrics;
trait AnnIndex<E: node::Element> {
    fn construct(&self); // construct algorithm structure
    fn add(&mut self, item: &node::Node<E>);
    fn once_constructed(&self) -> bool; // has already been constructed?
    fn reconstruct(&mut self);
    fn new(&self, dimension : usize, m: metrics::MetricType);

    fn search_k_node<F>(
        &self,
        item: &node::Node<E>,
        k: usize,
        metrics: &F,
    ) -> Vec<(node::Node<E>, E)>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>;

    fn search_k<F>(&self, is: &[E], k: usize, metrics: &F) -> Vec<(Vec<E>, E)>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>;
}
