use crate::core::heap::BinaryHeap;
use crate::core::metrics;
use crate::core::neighbor::Neighbor;
use crate::core::node::{FloatElement, IdxType, Node};
use fixedbitset::FixedBitSet;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub fn naive_build_knn_graph<E: FloatElement, T: IdxType>(
    nodes: &Vec<Box<Node<E, T>>>,
    mt: metrics::Metric,
    k: usize,
    graph: &mut Vec<Vec<Neighbor<E, usize>>>,
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
}

impl<'a, E: FloatElement, T: IdxType> NNDescentHandler<'a, E, T> {
    fn new(
        nodes: &'a Vec<Box<Node<E, T>>>,
        mt: metrics::Metric,
        k: usize,
        graph: &'a mut Vec<Vec<Neighbor<E, usize>>>,
        perturb_rate: f32,
        max_epoch: usize,
    ) -> Self {
        NNDescentHandler {
            nodes: nodes,
            graph: graph,
            mt: mt,
            k: k,
            visited_id: FixedBitSet::with_capacity(nodes.len() * nodes.len()),
            old_neighbors: Vec::new(),
            new_neighbors: Vec::new(),
            old_reversed_neighbors: HashMap::new(),
            new_reversed_neighbors: HashMap::new(),
            perturb_rate: perturb_rate,
            max_epoch: max_epoch,
        }
    }

    fn sample_neighbors(&mut self) {
        let nodes = &self.nodes;
        let mut visited_id = &mut self.visited_id;
        let mut graph = &mut self.graph;
        for n in 0..self.nodes.len() {
            graph[n].clear();
            for j in 0..self.k {
                let mut p = rand::thread_rng().gen_range(0, nodes.len());
                // while visited_id.contains(p + n * self.nodes.len()) && p != n {
                while p != n {
                    p = rand::thread_rng().gen_range(0, nodes.len());
                }
                graph[n].push(Neighbor::new(
                    p,
                    self.nodes[n].metric(&self.nodes[p], self.mt).unwrap(),
                ));
                // visited_id.insert(p + n * self.nodes.len());
            }
        }
    }

    fn sample_neighbors_set(&mut self, q: usize) {
        let mut sampled = Vec::new();
        let mut old_neighbors = Vec::new();
        let mut new_neighbors = Vec::new();
        let k = 5;

        let mut n = 0;
        for i in 0..self.graph[q].len() {
            if self.visited_id.contains(q * self.nodes.len() + i) {
                old_neighbors.push(self.graph[q][i].clone());
            } else {
                if n < k {
                    sampled.push(i);
                } else {
                    let mut m = rand::thread_rng().gen_range(0, k);
                    if m < k {
                        sampled.push(i);
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
        for epoch in 0..self.max_epoch {
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

        for i in 0..self.nodes.len() {
            for nb in self.old_neighbors[i].iter() {
                if !self.old_reversed_neighbors.contains_key(&nb.idx()) {
                    self.old_reversed_neighbors.insert(nb.idx(), Vec::new());
                }
                self.old_reversed_neighbors.get_mut(&nb.idx()).unwrap().push(nb.clone());
            }

            for nb in self.new_neighbors[i].iter() {
                if !self.new_reversed_neighbors.contains_key(&nb.idx()) {
                    self.new_reversed_neighbors.insert(nb.idx(), Vec::new());
                }
                self.new_reversed_neighbors.get_mut(&nb.idx()).unwrap().push(nb.clone());
            }
        }

        let mut update_nn = 0;
        for i in 0..self.nodes.len() {
            update_nn += self.local_join(i);
        }
        return update_nn;
    }

    fn local_join(&mut self, i: usize) -> usize {
        let k = 5;
        let mut update_count = 0;

        for iter in self.old_reversed_neighbors.get(&i).unwrap().iter() {
            let mut m = rand::thread_rng().gen_range(0, k);
            if rand::thread_rng().gen_range(0, k) < k {
                self.old_neighbors[i].push(iter.clone());
            }
        }

        for iter in self.new_reversed_neighbors.get(&i).unwrap().iter() {
            let mut m = rand::thread_rng().gen_range(0, k);
            if rand::thread_rng().gen_range(0, k) < k {
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
                    self.new_neighbors[i][q].idx(),
                    self.new_neighbors[i][p].idx(),
                );
                update_count += self.join(
                    self.new_neighbors[i][p].idx(),
                    self.new_neighbors[i][q].idx(),
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
        for j in 0..random_join {
            let mut nid = rand::thread_rng().gen_range(0, self.nodes.len());
            if nid >= i {
                nid += 1;
            }
            self.join(i, nid);
        }

        return update_count;
    }

    fn join(&mut self, me: usize, candidate: usize) -> usize {
        let s = self.nodes[me]
            .metric(&self.nodes[candidate], self.mt)
            .unwrap();
        if s < self.graph[me][self.graph[me].len() - 1].distance() {
            return 0;
        }

        let joiner_neighbor = Neighbor::new(candidate, s);
        let mut ub = 0;
        for i in 0..self.graph[me].len() {
            if self.graph[me][i].distance() < s {
                ub = i;
            }
        }

        let SHIFT = 20;
        let B = f32::MAX;
        let prB = self.perturb_rate * B;
        let rand_val = rand::thread_rng().gen_range(0, 100) as f32;
        if ub == self.graph[me].len()
            && self.graph[me][self.graph[me].len() - 1].distance() == s
            && rand_val > prB
        {
            return 0;
        }

        let mut lb = 0;
        for i in 0..self.graph[me].len() {
            if self.graph[me][i].distance() > s {
                lb = i;
            }
        }

        if self.graph[me].len() > 0 && self.graph[me][lb].distance() == s {
            for i in lb..ub {
                if self.graph[me][i].idx() == candidate {
                    return 0;
                }
            }
        }

        let pos = if lb < ub {
            lb + rand::thread_rng().gen_range(0, 100)
        } else {
            lb
        };

        if self.graph[me].len() < self.k {
            self.graph[me][pos] = joiner_neighbor;
            self.visited_id.set(me * self.nodes.len() + pos, false);
        } else {
            for i in pos..self.graph[me].len() {
                self.graph[me][i] = joiner_neighbor.clone();
                self.visited_id.set(me * self.nodes.len() + i, false);
            }
        }

        return 1;
    }
}
