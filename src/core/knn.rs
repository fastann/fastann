use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor::Neighbor;
use crate::core::node::{FloatElement, IdxType, Node};
use fixedbitset::FixedBitSet;
use rand::Rng;
use rayon::prelude::*;
use std::cmp;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub fn naive_build_knn_graph<E: FloatElement, T: IdxType>(
    nodes: &Vec<Box<Node<E, T>>>,
    mt: metrics::Metric,
    k: usize,
    graph: &mut Vec<Vec<Neighbor<E, usize>>>, // TODO: not use this one
) {
    let tmp_graph = Arc::new(Mutex::new(graph));
    (0..nodes.len()).into_par_iter().for_each(|n| {
        let item = &nodes[n];
        let mut heap = BinaryHeap::with_capacity(k);
        for i in 0..nodes.len() {
            if i == n {
                continue;
            }
            heap.push(Neighbor::new(i, item.metric(&nodes[i], mt).unwrap()));
            if heap.len() > k {
                heap.pop();
            }
        }
        let mut tmp = Vec::with_capacity(heap.len());
        while !heap.is_empty() {
            tmp.push(heap.pop().unwrap());
        }

        tmp_graph.lock().unwrap()[n].clear();
        tmp_graph.lock().unwrap()[n] = tmp;
    });
}

pub struct NNDescentHandler<'a, E: FloatElement, T: IdxType> {
    nodes: &'a Vec<Box<Node<E, T>>>,
    graph: &'a mut Vec<Vec<Neighbor<E, usize>>>,
    mt: metrics::Metric,
    k: usize,
    visited_id: FixedBitSet,
    old_neighbors: Vec<Vec<Neighbor<E, usize>>>,
    new_neighbors: Vec<Vec<Neighbor<E, usize>>>,
    old_reversed_neighbors: HashMap<usize, Vec<Neighbor<E, usize>>>,
    new_reversed_neighbors: HashMap<usize, Vec<Neighbor<E, usize>>>,
    perturb_rate: f32,
    max_epoch: usize,
    rho: f32,
}

impl<'a, E: FloatElement, T: IdxType> NNDescentHandler<'a, E, T> {
    fn new(
        nodes: &'a Vec<Box<Node<E, T>>>,
        mt: metrics::Metric,
        k: usize,
        graph: &'a mut Vec<Vec<Neighbor<E, usize>>>,
        perturb_rate: f32,
        max_epoch: usize,
        rho: f32,
    ) -> Self {
        NNDescentHandler {
            nodes,
            graph,
            mt,
            k,
            visited_id: FixedBitSet::with_capacity(nodes.len() * nodes.len()),
            old_neighbors: Vec::new(),
            new_neighbors: Vec::new(),
            old_reversed_neighbors: HashMap::new(),
            new_reversed_neighbors: HashMap::new(),
            perturb_rate,
            max_epoch,
            rho,
        }
    }

    fn random_neighbors(&mut self) {
        let nodes = &self.nodes;
        let _visited_id = &mut self.visited_id;
        let graph = &mut self.graph;
        let mut k = self.k;
        if k > self.nodes.len() - 1 {
            k = self.nodes.len() - 1;
        }
        for n in 0..self.nodes.len() {
            graph[n].clear();
            for _j in 0..k {
                let mut p = rand::thread_rng().gen_range(0, nodes.len());
                // while visited_id.contains(p + n * self.nodes.len()) && p != n {
                while p != n {
                    p = rand::thread_rng().gen_range(0, nodes.len());
                }
                graph[n].push(Neighbor::new(
                    p,
                    self.nodes[n].metric(&self.nodes[p], self.mt).unwrap(),
                ));
            }
            graph[n].sort();
        }
    }

    fn sample_neighbors_set(&mut self, q: usize) {
        let mut sampled = Vec::new();
        let mut old_neighbors = Vec::new();
        let mut new_neighbors = Vec::new();
        let rhk = cmp::min((self.rho * self.k as f32).ceil() as usize, self.nodes.len());

        let mut n = 0;
        for i in 0..self.graph[q].len() {
            if self.visited_id.contains(q * self.nodes.len() + i) {
                old_neighbors.push(self.graph[q][i].clone());
            } else {
                if n < rhk {
                    sampled.push(i);
                } else {
                    let m = rand::thread_rng().gen_range(0, n + 1);
                    if m < rhk {
                        sampled[m] = i;
                    }
                }
                n += 1;
            }
        }

        for i in 0..sampled.len() {
            self.visited_id.insert(q * self.nodes.len() + sampled[i]);
            new_neighbors.push(self.graph[q][i].clone());
        }
        self.new_neighbors.push(new_neighbors);
        self.old_neighbors.push(old_neighbors);
    }

    fn train(&mut self) {
        self.random_neighbors();
        for _epoch in 0..self.max_epoch {
            let update_count = self.update_graph();
            let kn = self.k * self.nodes.len();
            if (update_count as f32) <= 0.001 * (kn as f32) {
                break;
            }
        }
    }

    fn update_graph(&mut self) -> usize {
        (0..self.nodes.len()).for_each(|i| {
            self.sample_neighbors_set(i);
        });

        self.old_reversed_neighbors.clear();
        self.new_reversed_neighbors.clear();
        for i in 0..self.nodes.len() {
            self.old_reversed_neighbors.insert(i, Vec::new());
            self.new_reversed_neighbors.insert(i, Vec::new());
        }

        for i in 0..self.nodes.len() {
            for nb in self.old_neighbors[i].iter() {
                self.old_reversed_neighbors
                    .get_mut(&nb.idx())
                    .unwrap()
                    .push(Neighbor::new(i, nb.distance()));
            }

            for nb in self.new_neighbors[i].iter() {
                self.new_reversed_neighbors
                    .get_mut(&nb.idx())
                    .unwrap()
                    .push(Neighbor::new(i, nb.distance()));
            }
        }

        let mut update_nn = 0;
        for i in 0..self.nodes.len() {
            update_nn += self.local_join(i);
        }
        update_nn
    }

    fn local_join(&mut self, i: usize) -> usize {
        let rhk = cmp::min((self.rho * self.k as f32).ceil() as usize, self.nodes.len());
        let mut update_count = 0;

        for iter in self.old_reversed_neighbors.get(&i).unwrap().iter() {
            let _m =
                rand::thread_rng().gen_range(0, self.old_reversed_neighbors.get(&i).unwrap().len());
            if _m < rhk {
                self.old_neighbors[i].push(iter.clone());
            }
        }

        for iter in self.new_reversed_neighbors.get(&i).unwrap().iter() {
            let _m =
                rand::thread_rng().gen_range(0, self.new_reversed_neighbors.get(&i).unwrap().len());
            if _m < rhk {
                self.new_neighbors[i].push(iter.clone());
            }
        }

        for p in 0..self.new_neighbors[i].len() {
            update_count += self.join(self.new_neighbors[i][p].idx(), i);
        }

        for p in 0..self.new_neighbors[i].len() {
            for q in (p + 1)..self.new_neighbors[i].len() {
                if self.new_neighbors[i][p].idx() == self.new_neighbors[i][q].idx() {
                    continue;
                }
                update_count += self.join(
                    self.new_neighbors[i][p].idx(),
                    self.new_neighbors[i][q].idx(),
                );
                update_count += self.join(
                    self.new_neighbors[i][q].idx(),
                    self.new_neighbors[i][p].idx(),
                );
            }
        }

        for p in 0..self.new_neighbors[i].len() {
            for q in 0..self.old_neighbors[i].len() {
                if self.new_neighbors[i][p].idx() == self.old_neighbors[i][q].idx() {
                    continue;
                }
                update_count += self.join(
                    self.new_neighbors[i][p].idx(),
                    self.old_neighbors[i][q].idx(),
                );
                update_count += self.join(
                    self.old_neighbors[i][q].idx(),
                    self.new_neighbors[i][p].idx(),
                );
            }
        }

        let random_join = 10;
        for _j in 0..random_join {
            let mut nid = rand::thread_rng().gen_range(0, self.nodes.len() - 1);
            if nid >= i {
                nid += 1;
            }
            self.join(i, nid);
        }

        update_count
    }

    fn join(&mut self, me: usize, candidate: usize) -> usize {
        let s = self.nodes[me]
            .metric(&self.nodes[candidate], self.mt)
            .unwrap();
        if s < self.graph[me][self.graph[me].len() - 1].distance() && self.graph[me].len() == self.k
        {
            return 0;
        }

        let candidate_neighbor = Neighbor::new(candidate, s);
        let mut ub = self.graph[me].len() - 1;
        for i in 0..self.graph[me].len() {
            if self.graph[me][i].distance() > s {
                ub = i;
            }
        }

        let _SHIFT = 20;
        let B = 1 << _SHIFT;
        let prB = self.perturb_rate * (B as f32);
        let rand_val = rand::thread_rng().gen_range(0, B) as f32;
        if ub == self.graph[me].len()
            && self.graph[me][self.graph[me].len() - 1].distance() == s
            && rand_val > prB
        {
            return 0;
        }

        let mut lb = 0;
        for i in 0..self.graph[me].len() {
            if self.graph[me][i].distance() < s {
                lb = i;
            }
        }

        if !self.graph[me].is_empty() && self.graph[me][lb].distance() == s {
            for i in lb..ub {
                if self.graph[me][i].idx() == candidate {
                    return 0;
                }
            }
        }

        let pos = if lb < ub {
            lb + rand::thread_rng().gen_range(0, ub - lb)
        } else {
            lb
        };

        if self.graph[me].len() < self.k {
            self.new_neighbors[me][pos] = candidate_neighbor;
            self.visited_id.set(me * self.nodes.len() + pos, false);
        } else {
            for i in pos..self.graph[me].len() {
                if i < self.new_neighbors[me].len() {
                    self.new_neighbors[me][i] = candidate_neighbor.clone();
                } else {
                    self.new_neighbors[me].push(candidate_neighbor.clone());
                }

                self.visited_id.set(me * self.nodes.len() + i, false);
            }
        }

        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::node;
    use rand::distributions::{Distribution, Normal};
    use rand::Rng;
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
        let normal = Normal::new(0.0, range / 50.0);
        for _i in 0..clustering_n {
            let mut base: Vec<f64> = Vec::with_capacity(dimension);
            for _i in 0..dimension {
                let n: f64 = rng.gen_range(-range, range); // base number
                base.push(n);
            }

            for _i in 0..node_n {
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

        (bases, ns)
    }

    #[test]
    fn knn() {
        let dimension = 50;
        let nodes_every_cluster = 40;
        let node_n = 50;
        let (_, ns) =
            make_normal_distribution_clustering(node_n, nodes_every_cluster, dimension, 10000000.0);
        println!("hello world {:?}", ns.len());

        let mut data = Vec::new();
        for i in 0..ns.len() {
            data.push(Box::new(node::Node::new_with_idx(&ns[i], i)));
        }

        let mut graph: Vec<Vec<Neighbor<f64, usize>>> = vec![Vec::new(); data.len()];
        let base_start = SystemTime::now();
        naive_build_knn_graph::<f64, usize>(&data, metrics::Metric::DotProduct, 10, &mut graph);
        let base_since_the_epoch = SystemTime::now()
            .duration_since(base_start)
            .expect("Time went backwards");
        println!(
            "test for {:?} times, base use {:?} millisecond",
            ns.len(),
            base_since_the_epoch.as_millis()
        );

        let mut graph2: Vec<Vec<Neighbor<f64, usize>>> = vec![Vec::new(); data.len()];
        let base_start = SystemTime::now();
        let mut nn_descent_handler = NNDescentHandler::new(
            &data,
            metrics::Metric::DotProduct,
            10,
            &mut graph2,
            0.5,
            1,
            0.1,
        );
        nn_descent_handler.train();
        let base_since_the_epoch = SystemTime::now()
            .duration_since(base_start)
            .expect("Time went backwards");
        println!(
            "test for {:?} times, base use {:?} millisecond",
            ns.len(),
            base_since_the_epoch.as_millis()
        );

        let mut error = 0;
        for i in 0..graph.len() {
            for j in 0..graph[i].len() {
                if graph[i][j] != graph2[i][j] {
                    error += 1;
                }
            }
        }

        println!("error {}", error);
    }
}
