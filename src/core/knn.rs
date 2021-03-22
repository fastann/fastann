use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor::Neighbor;
use crate::core::node::{FloatElement, IdxType, Node};
use fixedbitset::FixedBitSet;
use rand::seq::SliceRandom;
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
    reversed_old_neighbors: Vec<Vec<usize>>,
    reversed_new_neighbors: Vec<Vec<usize>>,
    nn_old_neighbors: Vec<Vec<usize>>,
    nn_new_neighbors: Vec<Vec<usize>>,
    rho: f32,
    cost: usize,
    s: usize,
    update_cnt: usize,
}

impl<'a, E: FloatElement, T: IdxType> NNDescentHandler<'a, E, T> {
    fn new(
        nodes: &'a Vec<Box<Node<E, T>>>,
        mt: metrics::Metric,
        k: usize,
        graph: &'a mut Vec<Vec<Neighbor<E, usize>>>,
        rho: f32,
    ) -> Self {
        NNDescentHandler {
            nodes,
            graph,
            mt,
            k,
            visited_id: FixedBitSet::with_capacity(nodes.len() * nodes.len()),
            reversed_old_neighbors: Vec::new(),
            reversed_new_neighbors: Vec::new(),
            nn_new_neighbors: Vec::new(),
            nn_old_neighbors: Vec::new(),
            rho,
            cost: 0,
            s: (rho * k as f32) as usize,
            update_cnt: 0,
        }
    }

    fn update_nn_node(&mut self, me: usize, candidate: usize) -> i64 {
        let mut idx = self.graph[me].len() - 1;
        let mut j = 0;
        let dist = self.nodes[me]
            .metric(&self.nodes[candidate], self.mt)
            .unwrap();
        if dist > self.graph[me][self.graph[me].len() - 1].distance() {
            return -1;
        }
        loop {
            if idx == 0 {
                break;
            }
            j = idx - 1;
            if self.graph[me][j].idx() == candidate {
                return -1;
            }
            if self.graph[me][j].distance() < dist {
                break;
            }
            idx = j;
        }
        j = self.graph[me].len() - 1;
        loop {
            if j == idx {
                break;
            }
            self.graph[me][j] = self.graph[me][j - 1].clone();
            j -= 1;
        }
        self.graph[me][idx] = Neighbor::new(candidate, dist);
        idx as i64
    }

    fn update(&mut self, u1: usize, u2: usize) -> usize {
        if u1 == u2 {
            return 0;
        }

        if self.update_nn_node(u1, u2) != -1 {
            self.update_cnt += 1;
        }
        if self.update_nn_node(u2, u1) != -1 {
            self.update_cnt += 1;
        }
        self.visited_id.set(u1 * self.nodes.len() + u2, true);
        self.visited_id.set(u2 * self.nodes.len() + u1, true);
        1
    }

    fn init(&mut self) {
        self.visited_id = FixedBitSet::with_capacity(self.nodes.len() * self.nodes.len());
        self.graph.clear();
        for _i in 0..self.nodes.len() {
            let mut v = Vec::with_capacity(self.k);
            for _j in 0..self.k {
                v.push(Neighbor::new(self.nodes.len(), E::max_value()));
            }
            self.graph.push(v);
        }

        for i in 0..self.nodes.len() {
            self.nn_new_neighbors.push(Vec::new());
            self.nn_old_neighbors.push(Vec::new());
            for _j in 0..self.s {
                let rand_val = rand::thread_rng().gen_range(0, self.nodes.len());
                self.nn_new_neighbors[i].push(rand_val);
            }
        }

        for i in 0..self.nodes.len() {
            self.reversed_new_neighbors.push(Vec::new());
            self.reversed_old_neighbors.push(Vec::new());
            for _j in 0..self.s {
                let rand_val = rand::thread_rng().gen_range(0, self.nodes.len());
                self.reversed_new_neighbors[i].push(rand_val);
            }
        }
    }

    fn iterate(&mut self) -> usize {
        let mut cc = 0;
        self.update_cnt = 0;
        self.cost = 0;
        (0..self.nodes.len()).for_each(|i| {
            for j in 0..self.nn_new_neighbors[i].len() {
                for k in j..self.nn_new_neighbors[i].len() {
                    cc += self.update(self.nn_new_neighbors[i][j], self.nn_new_neighbors[i][k]);
                }
                for k in 0..self.nn_old_neighbors[i].len() {
                    cc += self.update(self.nn_new_neighbors[i][j], self.nn_old_neighbors[i][k]);
                }
            }

            for j in 0..self.reversed_new_neighbors[i].len() {
                for k in j..self.reversed_new_neighbors[i].len() {
                    if self.reversed_new_neighbors[i][j] >= self.reversed_new_neighbors[i][k] {
                        continue;
                    }
                    cc += self.update(
                        self.reversed_new_neighbors[i][j],
                        self.reversed_new_neighbors[i][k],
                    );
                }
                for k in 0..self.reversed_old_neighbors[i].len() {
                    cc += self.update(
                        self.reversed_new_neighbors[i][j],
                        self.reversed_old_neighbors[i][k],
                    );
                }
            }

            for j in 0..self.nn_new_neighbors[i].len() {
                for k in 0..self.reversed_old_neighbors[i].len() {
                    cc += self.update(
                        self.nn_new_neighbors[i][j],
                        self.reversed_old_neighbors[i][k],
                    );
                }
                for k in 0..self.reversed_new_neighbors[i].len() {
                    cc += self.update(
                        self.nn_new_neighbors[i][j],
                        self.reversed_new_neighbors[i][k],
                    );
                }
            }

            for j in 0..self.nn_old_neighbors[i].len() {
                for k in 0..self.reversed_new_neighbors[i].len() {
                    cc += self.update(
                        self.nn_old_neighbors[i][j],
                        self.reversed_new_neighbors[i][k],
                    );
                }
            }
        });

        self.cost += cc;
        let mut t = 0;

        for i in 0..self.nodes.len() {
            self.nn_new_neighbors[i].clear();
            self.nn_old_neighbors[i].clear();
            self.reversed_new_neighbors[i].clear();
            self.reversed_old_neighbors[i].clear();

            for j in 0..self.k {
                if self.graph[i][j].idx() == self.nodes.len() {
                    // init value, pass
                    continue;
                }
                if self
                    .visited_id
                    .contains(self.nodes.len() * i + self.graph[i][j].idx())
                {
                    self.nn_new_neighbors[i].push(j);
                } else {
                    self.nn_old_neighbors[i].push(self.graph[i][j].idx());
                }
            }

            t += self.nn_new_neighbors[i].len();

            if self.nn_new_neighbors[i].len() > self.s {
                let mut rng = rand::thread_rng();
                self.nn_new_neighbors[i].shuffle(&mut rng);
                self.nn_new_neighbors[i] = self.nn_new_neighbors[i][self.s..].to_vec();
            }

            for j in 0..self.nn_new_neighbors[i].len() {
                self.visited_id.set(
                    i * self.nodes.len() + self.graph[i][self.nn_new_neighbors[i][j]].idx(),
                    false,
                );
                self.nn_new_neighbors[i][j] = self.graph[i][self.nn_new_neighbors[i][j]].idx();
            }
        }

        (0..self.nodes.len()).for_each(|i| {
            for e in 0..self.nn_old_neighbors[i].len() {
                self.reversed_old_neighbors[self.nn_old_neighbors[i][e]].push(i);
            }
            for e in 0..self.nn_new_neighbors[i].len() {
                self.reversed_new_neighbors[self.nn_new_neighbors[i][e]].push(i);
            }
        });

        for i in 0..self.nodes.len() {
            if self.reversed_old_neighbors[i].len() > self.s {
                let mut rng = rand::thread_rng();
                self.reversed_old_neighbors[i].shuffle(&mut rng);
                self.reversed_old_neighbors[i] = self.reversed_old_neighbors[i][self.s..].to_vec();
            }
            if self.reversed_new_neighbors[i].len() > self.s {
                let mut rng = rand::thread_rng();
                self.reversed_new_neighbors[i].shuffle(&mut rng);
                self.reversed_new_neighbors[i] = self.reversed_new_neighbors[i][self.s..].to_vec();
            }
        }

        t
    }

    fn graph(&self) -> &Vec<Vec<Neighbor<E, usize>>> {
        self.graph
    }

    fn cost(&self) -> &usize {
        &self.cost
    }

    fn ths_update_cnt(&self) -> &usize {
        &self.update_cnt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::node;
    use rand::distributions::{Distribution, Normal};
    use rand::Rng;
    use std::collections::HashSet;
    use std::iter::FromIterator;
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
    fn knn_nn_descent() {
        let dimension = 2;
        let nodes_every_cluster = 200;
        let node_n = 20;
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
        let mut nn_descent_handler =
            NNDescentHandler::new(&data, metrics::Metric::DotProduct, 10, &mut graph2, 1.0);
        nn_descent_handler.init();

        let try_times = 50;
        let mut ground_truth: HashMap<usize, HashSet<usize>> = HashMap::new();
        for i in 0..graph.len() {
            ground_truth.insert(i, HashSet::from_iter(graph[i].iter().map(|x| x.idx())));
        }
        for _p in 0..try_times {
            let cc = nn_descent_handler.iterate();
            let mut error = 0;
            for i in 0..nn_descent_handler.graph.len() {
                for j in 0..nn_descent_handler.graph[i].len() {
                    if !ground_truth[&i].contains(&nn_descent_handler.graph[i][j].idx()) {
                        error += 1;
                    }
                }
            }
            println!(
                "error {} /{:?} cc {:?} cost {:?} update_cnt {:?}",
                error,
                data.len() * 10,
                cc,
                nn_descent_handler.cost(),
                nn_descent_handler.ths_update_cnt(),
            );
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
}
