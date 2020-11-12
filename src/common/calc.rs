extern crate num;

pub fn dot<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: Default + std::ops::Mul<Output = T> + std::ops::AddAssign + Copy,
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

pub fn manhanttan_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: Default + std::ops::Mul<Output = T> + std::ops::AddAssign + Copy + num::Signed,
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

pub fn enclidean_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: Default + std::ops::Mul<Output = T> + std::ops::AddAssign + Copy + num::Signed,
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

pub fn enclidean_distance_range<T>(vec1: &[T], vec2: &[T], begin: usize, end: usize) -> Result<T, &'static str>
where
    T: Default + std::ops::Mul<Output = T> + std::ops::AddAssign + Copy + num::Signed,
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
    T: Default + std::ops::Mul<Output = T> + std::ops::AddAssign + Copy + num::Float,
{
    match dot(&vec1, &vec1) {
        Ok(x) => x.sqrt(),
        _ => T::default(),
    }
}
