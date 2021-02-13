extern crate num;
use crate::core::calc::dot;
use crate::core::calc::same_dimension;
use crate::core::node::FloatElement;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Comparison {
    Bigger,
    Smaller,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Metric {
    Unknown,
    Manhattan,
    DotProduct,
    Euclidean,
    CosineSimilarity,
    AngularDistance,
}

impl Default for Metric {
    fn default() -> Self {
        Metric::Unknown
    }
}

// TODO: SIMD support
// TODO: make these func private
pub fn metric<T>(vec1: &[T], vec2: &[T], mt: Metric) -> Result<T, &'static str>
where
    T: FloatElement,
{
    return match mt {
        Metric::Euclidean => euclidean_distance(vec1, vec2),
        Metric::Manhattan => manhattan_distance(vec1, vec2),
        Metric::DotProduct => dot_product(vec1, vec2),
        Metric::CosineSimilarity => cosine_similarity(vec1, vec2),
        Metric::AngularDistance => angular_distance(vec1, vec2),
        Metric::Unknown => Result::Err("unknown method"),
    };
}

pub fn range_metric<T>(
    vec1: &[T],
    vec2: &[T],
    mt: Metric,
    begin: usize,
    end: usize,
) -> Result<T, &'static str>
where
    T: FloatElement,
{
    metric(&vec1[begin..end], &vec2[begin..end], mt)
}

pub fn dot_product<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    same_dimension(vec1, vec2)?;
    // smaller means closer.
    return match dot(vec1, vec2) {
        Ok(x) => Result::Ok(-x),
        Err(err) => Err(err),
    };
}

pub fn manhattan_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    same_dimension(vec1, vec2)?;
    let mut res = T::default();
    for i in 0..vec1.len() {
        res += vec1[i].abs() - vec2[i].abs();
    }
    return Result::Ok(res.abs());
}

pub fn euclidean_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    same_dimension(vec1, vec2)?;
    let mut res = T::default();
    for i in 0..vec1.len() {
        let diff = vec1[i] - vec2[i];
        res += diff * diff;
    }
    return Result::Ok(res.sqrt());
}

pub fn cosine_similarity<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    same_dimension(vec1, vec2)?;
    // smaller means closer.
    Result::Ok(
        -dot(vec1, vec2).unwrap()
            / (dot(vec1, vec1).unwrap().sqrt() * dot(vec2, vec2).unwrap().sqrt()),
    )
}

// (a/|a| - b/|b|)^2
// = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
// = 2 - 2cos
pub fn angular_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    same_dimension(vec1, vec2)?;
    let rhd = dot(vec1, vec1).unwrap();
    let lhd = dot(vec2, vec2).unwrap();
    let d = dot(vec1, vec1).unwrap();
    let m = rhd * lhd;
    let two = T::from_f32(2.0).unwrap();
    if m > T::float_zero() {
        Result::Ok(two - two * d / m.sqrt())
    } else {
        Result::Ok(two)
    }
}

pub fn euclidean_distance_range<T>(
    vec1: &[T],
    vec2: &[T],
    begin: usize,
    end: usize,
) -> Result<T, &'static str>
where
    T: FloatElement,
{
    same_dimension(vec1, vec2)?;
    euclidean_distance(&vec1[begin..end], &vec2[begin..end])
}
