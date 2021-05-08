use crate::core::calc::same_dimension;
#[cfg(feature = "simd")]
use packed_simd::{f32x4, f64x4};

pub trait SIMDOptmized<T = Self> {
    fn dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn manhattan_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn euclidean_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn miner_distance(a: &[T], b: &[T], c: &[T]) -> Result<(), &'static str>;
}

macro_rules! simd_optimized_impl {
    (  $type_id:ident, $simd_type:ident ,$size: expr ,$simd_size:expr) => {
        impl SIMDOptmized for $type_id {
            fn dot_product(a: &[$type_id], b: &[$type_id]) -> Result<$type_id, &'static str> {
                assert_eq!(a.len(), b.len());

                #[cfg(feature = $simd_size)]
                {
                    let size = 0;
                    let c: $type_id = 0.;
                    size = a.len() / $size;
                    c = a
                        .chunks_exact($size)
                        .map($simd_type::from_slice_unaligned)
                        .zip(b.chunks_exact($size).map($simd_type::from_slice_unaligned))
                        .map(|(a, b)| a * b)
                        .sum::<$simd_type>()
                        .sum();
                    let d: $type_id = (size..a.len()).map(|i| a[i] * b[i]).sum();
                    Ok(c + d)
                }
                #[cfg(not(feature = $simd_size))]
                {
                    Ok(a.iter().zip(b).map(|(p, q)| p * q).sum::<$type_id>())
                }
            }

            fn manhattan_distance(
                a: &[$type_id],
                b: &[$type_id],
            ) -> Result<$type_id, &'static str> {
                assert_eq!(a.len(), b.len());

                #[cfg(feature = $simd_size)]
                {
                    let size = 0;
                    let c: $type_id = 0.;
                    size = a.len() / $size;
                    c = a
                        .chunks_exact($size)
                        .map($simd_type::from_slice_unaligned)
                        .zip(b.chunks_exact($size).map($simd_type::from_slice_unaligned))
                        .map(|(a, b)| (a - b).abs())
                        .sum::<$simd_type>()
                        .sum();
                    let d: $type_id = (size..a.len()).map(|i| (a[i] - b[i]).abs()).sum();
                    Ok(c + d)
                }

                #[cfg(not(feature = $simd_size))]
                {
                    Ok(a.iter()
                        .zip(b)
                        .map(|(p, q)| (p - q).abs())
                        .sum::<$type_id>())
                }
            }

            fn euclidean_distance(
                a: &[$type_id],
                b: &[$type_id],
            ) -> Result<$type_id, &'static str> {
                same_dimension(a, b)?;

                #[cfg(feature = $simd_size)]
                {
                    let size = 0;
                    let c: $type_id = 0.;
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
                    let d: $type_id = (size..a.len()).map(|i| (a[i] - b[i]).powi(2)).sum();
                    Ok((d + c).sqrt())
                }
                #[cfg(not(feature = $simd_size))]
                {
                    Ok(a.iter()
                        .zip(b)
                        .map(|(p, q)| (p - q).powi(2))
                        .sum::<$type_id>()
                        .sqrt())
                }
            }

            fn miner_distance(a: &[$type_id], b: &[$type_id], c: &[$type_id]) -> Result<(), &'static str> {
                same_dimension(a, b)?;
                let size = 0;
                (0..a.len()).map(|i| c[i] = a[i] - b[i]);
                Ok(())
            }
        }
    };
}

simd_optimized_impl!(f32, f32x8, 8, "simd");
simd_optimized_impl!(f64, f64x4, 4, "simd");
