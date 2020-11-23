use crate::common::calc::manhanttan_distance;
use num::traits::NumAssign;

pub trait Element:
    Sized + Default + std::fmt::Debug + Clone + Copy + PartialEq + PartialOrd + NumAssign + num::Signed
{
}

pub trait FloatElement: Element + num::Float {}

#[macro_export]
macro_rules! to_element {
    (  $x:ident  ) => {
        impl Element for $x {}
    };
}

#[macro_export]
macro_rules! to_float_element {
    (  $x:ident  ) => {
        impl FloatElement for $x {}
    };
}

to_element!(f64);
to_element!(f32);
to_element!(i64);
to_element!(i32);
to_element!(i8);
to_element!(i16);
to_float_element!(f64);
to_float_element!(f32);

#[derive(Clone, Debug, Default)]
pub struct Node<E: Element> {
    vectors: Vec<E>,
    dimensions: usize,
}

impl<E: Element> Node<E> {
    pub fn new(vectors: &[E]) -> Node<E> {
        Node {
            vectors: vectors.to_vec(),
            dimensions: vectors.len(),
        }
    }

    pub fn distance<F>(&self, other: &Node<E>, cal: F) -> Result<E, &'static str>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>,
    {
        cal(&self.vectors, &other.vectors)
    }
}

#[cfg(test)]
#[test]
fn node_test() {
    // f64
    let v = vec![0.1, 0.2];
    let v2 = vec![0.2, 0.1];
    let n = Node::new(&v);
    let n2 = Node::new(&v2);
    n.distance(&n2, manhanttan_distance).unwrap();

    // int
    let v = vec![1, 1];
    let v2 = vec![2, 1];
    let n = Node::new(&v);
    let n2 = Node::new(&v2);
    n.distance(&n2, manhanttan_distance).unwrap();
}
