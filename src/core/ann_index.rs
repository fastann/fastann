use crate::core::metrics;
use crate::core::node;

pub trait AnnIndex<E: node::FloatElement, T: node::IdxType> {
    fn construct(&mut self) -> Result<(), &'static str>; // construct algorithm structure
    fn add(&mut self, item: &node::Node<E, T>) -> Result<(), &'static str>;
    fn add_with_vectors(&mut self, vs: &[E]) -> Result<(), &'static str> {
        let n = node::Node::new(vs);
        self.add(&n)
    }
    fn once_constructed(&self) -> bool; // has already been constructed?
    fn reconstruct(&mut self);
    fn node_search_k(
        &self,
        item: &node::Node<E, T>,
        k: usize,
        mt: metrics::Metric,
    ) -> Vec<(node::Node<E, T>, E)>;

    fn search_k(&self, item: &[E], k: usize, mt: metrics::Metric) -> Vec<(node::Node<E, T>, E)> {
        let n = node::Node::new(item);
        self.node_search_k(&n, k, mt)
    }

    fn load(&self, path: &str) -> Result<(), &'static str>;

    fn dump(&self, path: &str) -> Result<(), &'static str>;
}
