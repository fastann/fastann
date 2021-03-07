use packed_simd::f32x4;
use packed_simd::f64x4;

pub trait SIMDCalculable<T = Self> {
    fn dot_prod(a: &[T], b: &[T]) -> T;
}

impl SIMDCalculable for f32 {
    fn dot_prod(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        // assert!(a.len() % 4 == 0);

        let size = a.len() / 4;

        let mut c = a
            .chunks_exact(4)
            .map(f32x4::from_slice_unaligned)
            .zip(b.chunks_exact(4).map(f32x4::from_slice_unaligned))
            .map(|(a, b)| a * b)
            .sum::<f32x4>()
            .sum();
        for i in size..a.len() {
            c += a[i] * b[i];
        }
        c
    }
}

impl SIMDCalculable for f64 {
    fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        // assert!(a.len() % 4 == 0);

        let size = a.len() / 4;

        let mut c = a
            .chunks_exact(4)
            .map(f64x4::from_slice_unaligned)
            .zip(b.chunks_exact(4).map(f64x4::from_slice_unaligned))
            .map(|(a, b)| a * b)
            .sum::<f64x4>()
            .sum();
        for i in size*4..a.len() {
            c += a[i] * b[i];
        }
        c
    }
}
