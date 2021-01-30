extern crate num;
use crate::common::node::Element;
use crate::common::node::FloatElement;

// TODO: SIMD support

pub fn dot<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    if vec1.len() != vec2.len() {
        return Result::Err("different dimensions");
    }
    let mut res = T::default();
    for i in 0..vec1.len() {
        res += vec1[i] * vec2[i];
    }
    return Result::Ok(res);
}

pub fn manhattan_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: Element,
{
    if vec1.len() != vec2.len() {
        return Result::Err("different dimensions");
    }
    let mut res = T::default();
    for i in 0..vec1.len() {
        res += vec1[i].abs() - vec2[i].abs();
    }
    return Result::Ok(res);
}

pub fn euclidean_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: Element,
{
    if vec1.len() != vec2.len() {
        return Result::Err("different dimensions");
    }
    let mut res = T::default();
    for i in 0..vec1.len() {
        let diff = vec1[i] - vec2[i];
        res += diff * diff;
    }
    return Result::Ok(res);
}

pub fn euclidean_distance_range<T>(
    vec1: &[T],
    vec2: &[T],
    begin: usize,
    end: usize,
) -> Result<T, &'static str>
where
    T: Element,
{
    if vec1.len() != vec2.len() {
        return Result::Err("different dimensions");
    }
    let mut res = T::default();
    for i in begin..end {
        let diff = vec1[i] - vec2[i];
        res += diff * diff;
    }
    return Result::Ok(res);
}

pub fn get_norm<T>(vec1: &[T]) -> T
where
    T: FloatElement,
{
    match dot(&vec1, &vec1) {
        Ok(x) => x.sqrt(),
        _ => T::default(),
    }
}
