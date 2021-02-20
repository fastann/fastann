use crate::core::node::FloatElement;
use packed_simd_2;
use std::mem;

pub fn get_norm<T>(vec1: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    match dot(&vec1, &vec1) {
        Ok(val) => Ok(val.sqrt()),
        Err(err) => Err(err),
    }
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

#[macro_export]
macro_rules! dot_impl {
    ($type_id:ty, $length:expr, $simd_type:ty, $function_name:ident) => {
        fn $function_name(vec1: &[$type_id], vec2: &[$type_id]) -> Result<$type_id, &'static str> {
            same_dimension(vec1, vec2)?;
            Ok(vec1
                .chunks_exact($length)
                .map(<$simd_type>::from_slice_unaligned)
                .zip(
                    vec2.chunks_exact($length)
                        .map(<$simd_type>::from_slice_unaligned),
                )
                .map(|(a, b)| a * b)
                .sum::<$simd_type>()
                .sum())
        }
    };
}
dot_impl!(f64, 4, packed_simd_2::f64x4, dot_implementation_f64_4);
dot_impl!(f32, 4, packed_simd_2::f32x4, dot_implementation_f32_4);

#[macro_export]
macro_rules! dot_impl2 {
    ($type_id:ty, $length:expr, $simd_type:ty, $function_name:ident) => {
        pub fn dot<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
        where
            T: FloatElement,
        {
            same_dimension(vec1, vec2)?;
            Result::Ok(
                vec1.chunks_exact(8)
                    .map(f64x8::from_slice_unaligned)
                    .zip(vec2.chunks_exact(8).map(f64x8::from_slice_unaligned))
                    .map(|(a, b)| a * b)
                    .sum::<f64x8>()
                    .sum(),
            )
        }
    };
}

pub fn dot<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    same_dimension(vec1, vec2)?;
    let type_length = mem::size_of::<T>(); // f32 => 4, f64 => 8
    let mut res = T::default();
    for i in 0..vec1.len() {
        res += vec1[i] * vec2[i];
    }
    Result::Ok(res)
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
