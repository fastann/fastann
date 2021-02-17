use crate::core::arguments;
use crate::core::metrics;
use crate::core::node;

pub trait ANNIndex<E: node::FloatElement, T: node::IdxType> {
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
    fn once_constructed(&self) -> bool; // has already been constructed?
    fn reconstruct(&mut self, mt: metrics::Metric);
    fn node_search_k(
        &self,
        item: &node::Node<E, T>,
        k: usize,
        args: &arguments::Arguments,
    ) -> Vec<(node::Node<E, T>, E)>;

    fn search_k(&self, item: &[E], k: usize) -> Vec<(node::Node<E, T>, E)> {
        let n = node::Node::new(item);
        self.node_search_k(&n, k, &arguments::Arguments::new())
    }

    fn search_k_with_args(
        &self,
        item: &[E],
        k: usize,
        args: &arguments::Arguments,
    ) -> Vec<(node::Node<E, T>, E)> {
        let n = node::Node::new(item);
        self.node_search_k(&n, k, args)
    }

    fn load(&self, path: &str) -> Result<(), &'static str>;

    fn dump(&self, path: &str) -> Result<(), &'static str>;

    fn name(&self) -> &'static str;
}
