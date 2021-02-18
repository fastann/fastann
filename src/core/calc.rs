use crate::core::node::FloatElement;

pub fn get_norm<T>(vec1: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    match dot(&vec1, &vec1) {
        Ok(val) => Ok(val.sqrt()),
        Err(err) => Err(err),
    }
}

pub fn dot<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    same_dimension(vec1, vec2)?;
    let mut res = T::default();
    for i in 0..vec1.len() {
        res += vec1[i] * vec2[i];
    }
    Result::Ok(res)
}

pub fn same_dimension<T>(vec1: &[T], vec2: &[T]) -> Result<(), &'static str>
where
    T: FloatElement,
{
    if vec1.len() != vec2.len() {
        return Result::Err("different dimensions");
    }
    Result::Ok(())
}

pub fn split_imbalance<T>(vec1: &[T], vec2: &[T]) -> f64 {
    let ls = vec1.len() as f64;
    let rs = vec2.len() as f64;
    let f = ls / (ls + rs + 1e-9);
    if f > (1.0 - f) {
        f
    } else {
        1.0 - f
    }
}
