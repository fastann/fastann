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
    T::dot_product(&vec1, &vec2)
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "simd")]
    use crate::core::simd::Calculable;
    use rand::distributions::{Alphanumeric, StandardNormal, Uniform};
    use rand::distributions::{Distribution, Normal};
    use rand::seq::SliceRandom;
    use rand::{thread_rng, Rng};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    fn make_normal_distribution_clustering(
        clustering_n: usize,
        node_n: usize,
        dimension: usize,
        range: f64,
    ) -> (
        Vec<Vec<f64>>, // center of cluster
        Vec<Vec<f64>>, // cluster data
    ) {
        let mut rng = rand::thread_rng();

        let mut bases: Vec<Vec<f64>> = Vec::new();
        let mut ns: Vec<Vec<f64>> = Vec::new();
        let normal = Normal::new(0.0, (range / 50.0));
        for i in 0..clustering_n {
            let mut base: Vec<f64> = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let n: f64 = rng.gen_range(-range, range); // base number
                base.push(n);
            }

            for i in 0..node_n {
                let v_iter: Vec<f64> = rng.sample_iter(&normal).take(dimension).collect();
                let mut vec_item = Vec::with_capacity(dimension);
                for i in 0..dimension {
                    let vv = v_iter[i] + base[i]; // add normal distribution noise
                    vec_item.push(vv);
                }
                ns.push(vec_item);
            }
            bases.push(base);
        }

        return (bases, ns);
    }
    #[test]
    fn test_dot() {
        let a = [1., 2., 3.];
        let b = [1., 2., 3.];
        assert_eq!(dot(&a, &b).unwrap(), 14.0);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn bench_dot() {
        let dimension = 1024;
        let nodes_every_cluster = 40;
        let node_n = 50;
        let (_, ns) =
            make_normal_distribution_clustering(node_n, nodes_every_cluster, dimension, 10000000.0);
        println!("hello world {:?}", ns.len());

        {
            let base_start = SystemTime::now();
            for x in 0..ns.len() {
                dot(&ns[x], &ns[x]);
            }
            let base_since_the_epoch = SystemTime::now()
                .duration_since(base_start)
                .expect("Time went backwards");
            println!(
                "test for {:?} times, base use {:?} millisecond",
                ns.len(),
                base_since_the_epoch.as_millis()
            );
        }

        {
            let base_start = SystemTime::now();
            for x in 0..ns.len() {
                f64::dot_prod(&ns[x], &ns[x]);
                // println!("hello {:?}, {:?}", ns[x].len(), ns[x]);
            }
            let base_since_the_epoch = SystemTime::now()
                .duration_since(base_start)
                .expect("Time went backwards");
            println!(
                "test for {:?} times, base use {:?} millisecond",
                ns.len(),
                base_since_the_epoch.as_millis()
            );
        }

        let b = 25;
        println!(
            "{:?}, {:?}",
            f64::dot_prod(&ns[b], &ns[b]),
            dot(&ns[b], &ns[b]).unwrap()
        );
    }
}
