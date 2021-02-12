use crate::core::node::FloatElement;
use crate::core::metrics::dot;

pub fn get_norm<T>(vec1: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    return match dot(&vec1, &vec1) {
        Ok(val) => Ok(val.sqrt()),
        Err(err) => Err(err),
    };
}