use crate::core::metrics;
use crate::core::metrics::manhattan_distance;
use num::traits::{FromPrimitive, NumAssign};
use std::fmt::Display;

pub trait FloatElement:
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
    + num::Float
{
    fn float_one() -> Self {
        return Self::from_f32(1.0).unwrap();
    }

    fn float_zero() -> Self {
        return Self::from_f32(0.0).unwrap();
    }

    fn zero_patch_num() -> Self {
        return Self::from_f32(1.34e-6_f32).unwrap();
    }
}

pub trait IdxType: Sized + Clone + Default + std::fmt::Debug + Eq + Ord {}

#[macro_export]
macro_rules! to_float_element {
    (  $x:ident  ) => {
        impl FloatElement for $x {}
    };
}

#[macro_export]
macro_rules! to_idx_type {
    (  $x:ident  ) => {
        impl IdxType for $x {}
    };
}

to_float_element!(f64);
to_float_element!(f32);

to_idx_type!(String);
to_idx_type!(usize);
to_idx_type!(i16);
to_idx_type!(i32);
to_idx_type!(i64);
to_idx_type!(i128);
to_idx_type!(u16);
to_idx_type!(u32);
to_idx_type!(u64);
to_idx_type!(u128);

#[derive(Clone, Debug, Default)]
pub struct Node<E: FloatElement, T: IdxType> {
    vectors: Vec<E>, // the vectors;
    idx: Option<T>, // data id, it can be any type;
}

impl<E: FloatElement, T: IdxType> Node<E, T> {
    // TODO: make it Result
    pub fn new(vectors: &[E]) -> Node<E, T> {
        vectors.iter().map(|x| {
            if Node::<E, T>::valid_elements(x) {
                // TODO: do somthing
            }
        });
        Node {
            vectors: vectors.to_vec(),
            idx: Option::None,
        }
    }

    pub fn new_with_idx(vectors: &[E], id: T) -> Node<E, T> {
        Node {
            vectors: vectors.to_vec(),
            idx: Option::Some(id),
        }
    }

    pub fn distance<F>(&self, other: &Node<E, T>, calc: F) -> Result<E, &'static str>
    where
        F: Fn(&[E], &[E]) -> Result<E, &'static str>,
    {
        calc(&self.vectors, &other.vectors)
    }

    pub fn metric(&self, other: &Node<E, T>, t: metrics::MetricType) -> Result<E, &'static str> {
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

    pub fn idx(&self) -> Option<T> {
        match &self.idx {
            Some(k) => Some(k.clone()),
            None => None,
        }
    }

    fn valid_elements(e: &E) -> bool {
        e.is_nan() || e.is_infinite() || !e.is_normal()
    }
}

impl<E: FloatElement, T: IdxType> std::fmt::Display for Node<E, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "(key: {:#?}, vectors: {:#?})", self.idx, self.vectors)
    }
}

// general method

#[cfg(test)]
#[test]
fn node_test() {
    // f64
    let v = vec![0.1, 0.2];
    let v2 = vec![0.2, 0.1];
    let n = Node::<f64, usize>::new(&v);
    let n2 = Node::<f64, usize>::new(&v2);
    n.distance(&n2, manhattan_distance).unwrap();
}
