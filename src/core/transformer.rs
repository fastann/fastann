use crate::core::metrics;
use crate::core::simd_metrics;
use crate::core::node;

use core::hash::Hash;
use core::iter::Sum;
use cblas::*;


pub struct Linear {
    _dim_in: usize,
    _dim_out: usize,
    _has_bias, bool,
    _is_trained: bool,
    _W: Vec<f32>, // size _dim_in * _dim_out
    _b: Vec<f32>, // size _dim_out
}

impl Linear{
    pub fn new() -> Linear{
        Linear(dim_in: usize, dim_out: usize, has_bias: bool){
            _dim_in: dim_in,
            _dim_out: dim_out,
            _has_bias: has_bias,
            _is_trained: false, // will be trained when W and b are initialized
            ..Default::default()
        }
    }
    
    // y = W * x + b
    pub fn forward(n:usize, Vec<f32> x){
        let mut y: Vec<f32> = Vec::new();
    }

}
