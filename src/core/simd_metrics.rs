use crate::core::calc::same_dimension;
#[cfg(feature = "simd")]
use packed_simd::{f32x4, f64x4};

pub trait SIMDOptmized<T = Self> {
    fn dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn manhattan_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn euclidean_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
}

macro_rules! simd_optimized_impl {
    (  $type_id:ident, $simd_type:ident ,$size: expr ,$simd_size:expr) => {
        impl SIMDOptmized for $type_id {
            fn dot_product(a: &[$type_id], b: &[$type_id]) -> Result<$type_id, &'static str> {
                same_dimension(a, b)?;
                let mut size = 0;
                let mut c: $type_id = 0.;
                #[cfg(feature = $simd_size)]
                {
                    size = a.len() / $size;
                    c = a
                        .chunks_exact($size)
                        .map($simd_type::from_slice_unaligned)
                        .zip(b.chunks_exact($size).map($simd_type::from_slice_unaligned))
                        .map(|(a, b)| a * b)
                        .sum::<$simd_type>()
                        .sum();
                }
                for i in size..a.len() {
                    c += a[i] * b[i];
                }
                Ok(c)
            }

            fn manhattan_distance(
                a: &[$type_id],
                b: &[$type_id],
            ) -> Result<$type_id, &'static str> {
                same_dimension(a, b)?;
                let mut size = 0;
                let mut c: $type_id = 0.;
                #[cfg(feature = $simd_size)]
                {
                    size = a.len() / $size;
                    c = a
                        .chunks_exact($size)
                        .map($simd_type::from_slice_unaligned)
                        .zip(b.chunks_exact($size).map($simd_type::from_slice_unaligned))
                        .map(|(a, b)| (a - b).abs())
                        .sum::<$simd_type>()
                        .sum();
                }
                for i in size..a.len() {
                    c += (a[i] - b[i]).abs();
                }
                Ok(c)
            }

            fn euclidean_distance(
                a: &[$type_id],
                b: &[$type_id],
            ) -> Result<$type_id, &'static str> {
                same_dimension(a, b)?;
                let mut size = 0;
                let mut c: $type_id = 0.;
                #[cfg(feature = $simd_size)]
                {
                    size = a.len() / $size;
                    c = a
                        .chunks_exact($size)
                        .map($simd_type::from_slice_unaligned)
                        .zip(b.chunks_exact($size).map($simd_type::from_slice_unaligned))
                        .map(|(a, b)| {
                            let c = (a - b);
                            c * c
                        })
                        .sum::<$simd_type>()
                        .sum();
                }
                for i in size..a.len() {
                    c += (a[i] - b[i]).powi(2);
                }
                Ok(c)
            }
        }
    };
}

simd_optimized_impl!(f32, f32x8, 8, "simd");
simd_optimized_impl!(f64, f64x4, 4, "simd");
