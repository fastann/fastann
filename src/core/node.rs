use crate::core::metrics;
use crate::core::metrics::manhattan_distance;
use num::traits::{FromPrimitive, NumAssign};
use std::fmt::Display;

trait Element:
    FromPrimitive
    + Sized
    + Default
    + num::Zero
    + std::fmt::Debug
    + Clone
    + Copy
    + PartialEq
    + PartialOrd
    + NumAssign
    + num::Signed
{
    fn int_one() -> Self {
        return Self::from_i8(1).unwrap();
    }

    fn int_zero() -> Self {
        return Self::from_i8(1).unwrap();
    }
}

pub trait FloatElement: Element + num::Float {
    fn float_one() -> Self {
        return Self::from_f32(1.0).unwrap();
    }

    fn float_zero() -> Self {
        return Self::from_f32(0.0).unwrap();
    }
}

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
to_float_element!(f64);
to_float_element!(f32);

#[derive(Clone, Debug, Default)]
pub struct Node<E: FloatElement> {
    vectors: Vec<E>,
    id: Option<usize>,
}

impl<E: FloatElement> Node<E> {
    // TODO: make it Result
    pub fn new(vectors: &[E]) -> Node<E> {
        vectors.iter().map(|x| {
            if (Node::valid_elements(x)) {
                // TODO: do somthing
            }
        });
        Node {
            vectors: vectors.to_vec(),
            id: Option::None,
        }
    }

    pub fn new_with_id(vectors: &[E], id: usize) -> Node<E> {
        Node {
            vectors: vectors.to_vec(),
            id: Option::Some(id),
        }
    }

    pub fn distance<F>(&self, other: &Node<E>, cal: F) -> Result<E, &'static str>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>,
    {
        cal(&self.vectors, &other.vectors)
    }

    pub fn metric(&self, other: &Node<E>, t: &metrics::MetricType) -> Result<E, &'static str> {
        metrics::metric(&self.vectors, &other.vectors, t)
    }

    // const value
    pub fn vectors(&self) -> &Vec<E> {
        &self.vectors
    }

    pub fn mut_vectors(&mut self) -> &mut Vec<E> {
        &mut self.vectors
    }

    pub fn set_vectors(&mut self, v: &[E]) {
        self.vectors = v.to_vec();
    }

    pub fn push(&mut self, e: &E) {
        self.vectors.push(e.clone());
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn id(&self) -> Option<usize> {
        self.id
    }

    fn valid_elements(e: &E) -> bool {
        e.is_nan() || e.is_infinite() || !e.is_normal()
    }
}

impl<E: FloatElement> std::fmt::Display for Node<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "(id: {:#?}, vectors: {:#?})", self.id, self.vectors)
    }
}

// general method

#[cfg(test)]
#[test]
fn node_test() {
    // f64
    let v = vec![0.1, 0.2];
    let v2 = vec![0.2, 0.1];
    let n = Node::new(&v);
    let n2 = Node::new(&v2);
    n.distance(&n2, manhattan_distance).unwrap();
}
