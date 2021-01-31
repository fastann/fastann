use crate::core::metrics;
use crate::core::node;

#[derive(Default)]
pub struct Parameters {
    dimension: usize,       // dimension
    m: metrics::MetricType, // metric type
}