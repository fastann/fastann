use crate::common::node;
trait AnnIndex<E: node::Element> {
    fn construct(&self); // construct algorithm structure
    fn add(&mut self, item: &node::Node<E>);
    fn lazy_add(&mut self, item: &node::Node<E>);
    fn once_constructed(&self) -> bool;

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
