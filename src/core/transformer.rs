use crate::core::metrics;
use crate::core::simd_metrics;
use crate::core::node;

use core::hash::Hash;
use core::iter::Sum;
use cblas::*;

use ndarray::{ArrayBase, Array2, Array1, Axis, Data, Ix2, arr2, stack, s, Zip};
use ndarray_linalg::{SVD, Trace};
use std::cmp::max;
extern crate blas;
extern crate openblas_src;



#[derive(Default, Debug)]
pub struct Linear {
    _dim_in: usize,
    _dim_out: usize,
    _has_bias: bool,
    _is_trained: bool,
    _W: Array2<f64>, // size _dim_in * _dim_out
    _b: Array1<f64>, // size _dim_out
}

impl Linear{
    pub fn new(dim_in: usize, dim_out: usize, has_bias: bool) -> Linear{
        Linear{
            _dim_in: dim_in,
            _dim_out: dim_out,
            _has_bias: has_bias,
            _is_trained: false, // will be trained when W and b are initialized
            ..Default::default()
        }
    }
    
    pub fn init(&mut self, W: Array2<f64>, b: Array1<f64>){
        self._W = W;
        self._b = b;
        self._is_trained = true;
    }

    // y = W * x + b
    pub fn forward(&self, n:usize,  x: Array2<f64>) -> Result<Array2<f64>,  &'static str>{
        if self._has_bias && self._b.len() != self._dim_out{
            return Err("Wrong _b len");
        }

        if self._W.shape()[1] != self._dim_out && self._W.shape()[0] != self._dim_in {
            return Err("Wrong _W len");
        }

        let h:Array2<f64> = Array2::ones((x.shape()[0], self._dim_out));
        let y = x.dot(&self._W) + h.dot(&self._b);

        return Ok(y);
    }

}

#[derive(Default, Debug)]
pub struct PCA {
    _dim_in: usize,
    _dim_out: usize,
    _components: Array2<f64>,
    _mean : Array1<f64>,
    _W: Array2<f64>,
}

impl PCA {

    pub fn new() -> PCA{
        PCA{
            _dim_in: 0,
            _dim_out: 0,
            ..Default::default()
        }
    }

    pub fn fit (
        &mut self,
        x : &ArrayBase<impl Data<Elem = f64>, Ix2>,
        n_components : f64,
    ) {

        let (_n, _m) = x.dim();
        //calculate the array of columnar means
        let mean = x.mean_axis(Axis(0)).unwrap();

        // subtract means from X
        let h:Array2<f64> = Array2::ones((_n, _m));
        let temp:Array2<f64> = h * &mean;
        let b:Array2<f64> = x - &temp;

        // compute SVD
        let (u, sigma, v) = b.svd(true, true).unwrap();

        let mut u = u.unwrap() as Array2<f64>;
        let mut v = v.unwrap() as Array2<f64>;
        let temp = (b.nrows() - 1) as f64;
        let explained_variance = sigma.map(|x| x.powi(2)).map(|x|  x / temp );
        let total_var = explained_variance.sum();
        let explained_variance_ratio = explained_variance.map(|x| x / total_var);
        let mut singular_values = sigma.clone();

        let mut ratio_cumsum = explained_variance_ratio.clone();
        ratio_cumsum.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

        //println!("{}", ratio_cumsum);
        // find the number of components to represent the variance ration passed in as n_components_ratio
        let mut dim_out = 0;
        if n_components < 1.0 {
            let mut covered_components = 0;
            loop {
                if ratio_cumsum[covered_components] > n_components {
                    covered_components += 1;
                    break;
                } else {
                    covered_components += 1;
                }
            }
            v = v.t().slice(s![.., ..covered_components]).to_owned();
            u = u.slice(s![.., ..covered_components]).to_owned();
            singular_values = singular_values.slice(s![..covered_components]).to_owned();
            dim_out = covered_components;
        }
        else if n_components == 1.0 {
            v = v.t().slice(s![.., ..]).to_owned();
            u = u.slice(s![.., ..]).to_owned();
            singular_values = singular_values.slice(s![..]).to_owned();
            dim_out = _m;
        }
        else {
            v = v.t().slice(s![.., ..n_components as i32]).to_owned();
            u = u.slice(s![.., ..n_components as i32]).to_owned();
            singular_values = singular_values.slice(s![..n_components as i32]).to_owned();
            dim_out = n_components as usize;
        }

        // println!("sigma: {:?}", sigma);
        // println!("singular_values: {:?}", singular_values);
        // println!("y: {:?}", b.dot(&v));
        let components = u * singular_values;
        self._dim_in = _m;
        self._dim_out = dim_out;
        self._components = components;
        self._mean = mean;
        self._W = v;
        
        // println!("components: {:?}", self._components);
        // println!("mean: {:?}", self._mean);
    }

    pub fn forward(&self, x: Array2<f64>) -> Result< Array2<f64>, &'static str>{
        if x.shape()[1] != self._dim_in{
            return Err("pca dim error");
        }
        return Ok(x.dot(&self._W));
    }

    pub fn mean(&self) -> &Array1<f64> {
        &self._mean
    }

    pub fn components(&self) -> &Array2<f64> {
        &self._components
    }
}
