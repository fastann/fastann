use crate::annoy::def;
use crate::annoy::random;
use crate::common::calc;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;

// TODO: node as a trait with getter setter function
#[derive(Default, Clone, PartialEq, Debug)]
pub struct Node {
    n_descendants: i32, // tot n_descendants
    children: Vec<i32>, // left and right and if it's a leaf node, children would be very large (depend on _K)
    v: Vec<f64>,

    // biz field
    // TODO: use trait
    norm: f64,

    has_init: bool,
}

impl Node {
    fn new() -> Node {
        Node {
            children: vec![0, 0],
            ..Default::default()
        }
    }

    fn new_with_vectors(_v: &[f64]) -> Node {
        Node {
            children: vec![0, 0],
            v: _v.to_vec(),
            ..Default::default()
        }
    }

    fn is_empty(&self) -> bool {
        return self.has_init;
    }

    fn init(&mut self) {
        self.children = vec![0, 0];
    }

    fn copy(dst: &mut Node, src: &Node) {
        dst.n_descendants = src.n_descendants.clone();
        dst.children = src.children.clone();
        dst.v = src.v.clone();
        dst.norm = src.norm.clone();
    }
}

#[derive(Default, Clone, PartialEq, Debug)]
struct Neighbor {
    idx: i32,
    distance: f64,
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Neighbor) -> Ordering {
        let ord = if self.distance > other.distance {
            Ordering::Greater
        } else if self.distance < other.distance {
            Ordering::Less
        } else {
            Ordering::Equal
        };
        return ord;
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Neighbor) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Neighbor {}

pub fn two_means<D: Distance + Base>(
    nodes: &[Node],
    use_cosine: bool,
    distance: &D,
) -> Result<(Node, Node), &'static str> {
    if nodes.len() < 2 {
        return Err("empty nodes");
    }
    let count = nodes.len();
    let i = random::index(count);
    let mut j = random::index(count - 1);

    if j >= i {
        j += 1;
    }

    let mut p = distance.copy_node(&nodes[i]);
    let mut q = distance.copy_node(&nodes[j]);

    if use_cosine {
        distance.normalize(&mut p);
        distance.normalize(&mut q);
    }
    distance.init_node(&mut p);
    distance.init_node(&mut q);

    let mut ic: f64 = 1.0;
    let mut jc: f64 = 1.0;

    for _z in 0..def::ITERATION_STEPS {
        let k = random::index(count);
        let di = ic * distance.distance(&p, &nodes[k])?;
        let dj = jc * distance.distance(&q, &nodes[k])?;
        let norm = if use_cosine {
            calc::get_norm(&nodes[k].v)
        } else {
            1.0
        };

        if !(norm > 0.0) {
            continue;
        }

        if di < dj {
            for l in 0..q.v.len() {
                p.v[l] = (p.v[l] * ic + nodes[k].v[l] / norm) / (ic + 1.0);
            }
            distance.init_node(&mut p);
            ic += 1.0;
        } else {
            for l in 0..q.v.len() {
                q.v[l] = (q.v[l] * ic + nodes[k].v[l] / norm) / (ic + 1.0);
            }
            distance.init_node(&mut q);
            jc += 1.0;
        }
    }

    return Ok((p, q));
}

pub trait Base: Default {
    // TODO:
    fn preprocess(&self, nodes: &[Node]) {}

    fn zero_value(&self, src: &mut Node) {}
    fn copy_node(&self, src: &Node) -> Node {
        return Node {
            n_descendants: src.n_descendants.clone(),
            v: src.v.clone(),
            children: src.children.clone(),
            ..Default::default()
        };
    }
    fn normalize(&self, node: &mut Node) {
        let norm = calc::get_norm(&node.v);
        if norm > 0.0 {
            for i in 0..node.v.len() {
                node.v[i] /= norm;
            }
        }
    }
}

pub trait Distance {
    fn init_node(&self, node: &mut Node) {}
    fn distance(&self, src: &Node, dst: &Node) -> Result<f64, &'static str>;
    fn create_split(&self, nodes: &[Node], n: &mut Node) -> Result<(), &'static str>;
    fn pq_initial_value(&self) -> f64 {
        return f64::MAX;
    }
    fn side(&self, src: &Node, dst: &[f64]) -> bool {
        return false;
    }

    fn margin(&self, src: &Node, dst: &[f64]) -> Result<f64, &'static str> {
        return Ok(0.0);
    }

    fn pq_distance(&self, distance: f64, mut margin: f64, child_nr: usize) -> f64 {
        return 0.0;
    }
}

#[derive(Default, Clone, Debug, Copy)]
pub struct Angular {}

impl Base for Angular {
    fn copy_node(&self, src: &Node) -> Node {
        return src.clone();
    }
}

impl Distance for Angular {
    fn distance(&self, src: &Node, dst: &Node) -> Result<f64, &'static str> {
        let left = if src.norm != 0.0 {
            src.norm
        } else {
            calc::dot(&src.v, &src.v)?
        };
        let right = if dst.norm != 0.0 {
            dst.norm
        } else {
            calc::dot(&dst.v, &dst.v)?
        };
        let dot_val = calc::dot(&src.v, &dst.v)?;
        let inner_val = right * left;
        if inner_val > 0.0 {
            return Result::Ok(2.0 - 2.0 * dot_val / inner_val.sqrt());
        } else {
            return Result::Ok(2.0);
        }
    }

    fn margin(&self, src: &Node, dst: &[f64]) -> Result<f64, &'static str> {
        return calc::dot(&src.v, &dst);
    }

    fn side(&self, src: &Node, dst: &[f64]) -> bool {
        match self.margin(&src, &dst) {
            Ok(x) => {
                return x > 0.0;
            }
            Err(e) => {
                return random::flip();
            }
        }
    }

    fn create_split(&self, nodes: &[Node], n: &mut Node) -> Result<(), &'static str> {
        let (p, q) = two_means(&nodes, true, self)?;

        if n.v.len() != 0 && n.v.len() != p.v.len() {
            return Err("empty node input");
        }

        let is_initial = if n.v.len() == 0 { true } else { false };
        for i in 0..p.v.len() {
            if is_initial {
                n.v.push(p.v[i] - q.v[i]);
            } else {
                n.v[i] = p.v[i] - q.v[i];
            }
        }
        self.normalize(n);
        return Ok(());
    }

    fn init_node(&self, node: &mut Node) {
        match calc::dot(&node.v, &node.v) {
            Ok(dot) => {
                node.norm = dot;
            }
            Err(e) => return, // do nothing
        }
    }

    fn pq_distance(&self, distance: f64, mut margin: f64, child_nr: usize) -> f64 {
        if child_nr == 0 {
            margin = -margin;
        }
        if distance < margin {
            return distance;
        } else {
            return margin;
        }
    }
}

impl Angular {
    fn normalize_distances(&self, distance: f64) -> f64 {
        if distance > 0.0 {
            return distance.sqrt();
        }
        return 0.0;
    }

    fn pq_initial_value(&self) -> f64 {
        return f64::MAX;
    }

    fn name(&self) -> &'static str {
        "angular"
    }
}

#[derive(Default, Clone, Debug, Copy)]
pub struct DotProduct {
    angular: Angular,
}

impl Distance for DotProduct {
    fn distance(&self, src: &Node, dst: &Node) -> Result<f64, &'static str> {
        return Ok(-calc::dot(&src.v, &dst.v)?);
    }

    fn create_split(&self, nodes: &[Node], n: &mut Node) -> Result<(), &'static str> {
        return Ok(self.angular.create_split(&nodes, n)?);
    }

    fn init_node(&self, node: &mut Node) {}
}

impl Base for DotProduct {}

// TODO: implement
impl DotProduct {}

pub trait AnnoyIndexer<D: Distance + Base> {
    fn add_item(&mut self, item: i32, w: &[f64], d: D) -> Result<(), &'static str>;
    fn build(&mut self, q: i32) -> Result<(), &'static str>;
    fn unbuild(&mut self) -> Result<(), &'static str> {
        return Ok(());
    }
    fn save(&self, filename: &str) -> Result<(), &'static str> {
        return Ok(());
    }
    fn load(&self) -> Result<(), &'static str> {
        return Ok(());
    }
    fn get_distance(&self, i: i32, j: i32) -> Result<f64, &'static str> {
        return Ok(0.0);
    }
    fn get_nns_by_item(
        &self,
        idx: i32,
        n: usize,
        search_k: i32,
    ) -> Result<(Vec<i32>, Vec<f64>), &'static str> {
        return Err("please implement the method");
    }
    fn get_nns_by_vector(
        &self,
        f: &[f64],
        n: usize,
        search_k: i32,
    ) -> Result<(Vec<i32>, Vec<f64>), &'static str> {
        return Err("please implement the method");
    }
    fn get_n_items(&self) -> i32 {
        return 0;
    }
    fn get_n_tree(&self) -> i32 {
        return 0;
    }
    fn verbose(&self, v: bool) {}
    fn get_item(&self, item: f64) -> Result<f64, &'static str> {
        return Ok(0.0);
    }
    fn set_seed(&mut self, q: i32) {}
    // fn on_disk_build
}

#[derive(Default, Debug)]
pub struct AnnoyIndex<D: Distance + Base> {
    _f: usize, // dimension
    // _s: i32,       // node size
    _n_items: i32, // add items count
    // _nodes;
    _n_nodes: i32, // nodes count
    // _nodes_size: i32, // in source code, this means the memory which has been allocated, and we can use node's size to get data
    _roots: Vec<i32>, // dummy root's children
    _K: i32,          // max number of n_descendants to fit into node
    _is_seeded: bool,
    _seed: i32,
    _loaded: bool,
    _verbose: bool,
    _fd: i32,
    _on_disk: bool,
    _built: bool,
    pub nodes: Vec<Node>,

    distance: D,
}

impl<D: Distance + Base> AnnoyIndexer<D> for AnnoyIndex<D> {
    fn add_item(&mut self, item: i32, w: &[f64], d: D) -> Result<(), &'static str> {
        if w.len() != self._f {
            return Err("dimension is different");
        }
        if self._loaded {
            return Err("you can't add an item to a loaded index");
        }

        let mut nn = Node::new();

        d.zero_value(&mut nn);

        nn.children[0] = 0;
        nn.children[1] = 0;
        nn.n_descendants = 1; // only the node itself

        for i in 0..self._f {
            nn.v.push(w[i]);
        }

        d.init_node(&mut nn);

        self._n_items += 1;

        self.nodes.push(nn);

        return Ok(());
    }

    fn build(&mut self, q: i32) -> Result<(), &'static str> {
        if self._built {
            return Err("has built");
        }

        self.distance.preprocess(&self.nodes);

        self._n_nodes = self._n_items;
        self.thread_build(q);
        self._built = true;
        return Ok(());
    }

    fn unbuild(&mut self) -> Result<(), &'static str> {
        self._roots.clear();
        self._n_nodes = self._n_items;
        self._built = false;
        return Ok(());
    }
    fn get_distance(&self, i: i32, j: i32) -> Result<f64, &'static str> {
        let ni = match self.get_node(i) {
            Some(node) => node,
            None => return Err("not existing"),
        };
        let nj = match self.get_node(j) {
            Some(node) => node,
            None => return Err("not existing"),
        };
        return Ok(self.distance.distance(&ni, &nj)?);
    }

    fn get_nns_by_item(
        &self,
        idx: i32,
        n: usize,
        search_k: i32,
    ) -> Result<(Vec<i32>, Vec<f64>), &'static str> {
        match self.get_node(idx) {
            Some(node) => self.get_all_nns(&node.v, n, search_k),
            None => return Err("invalid idx"),
        }
    }

    fn get_nns_by_vector(
        &self,
        f: &[f64],
        n: usize,
        search_k: i32,
    ) -> Result<(Vec<i32>, Vec<f64>), &'static str> {
        self.get_all_nns(&f, n, search_k)
    }

    fn get_n_items(&self) -> i32 {
        return self._n_items;
    }
    fn get_n_tree(&self) -> i32 {
        return self._roots.len() as i32;
    }

    fn set_seed(&mut self, q: i32) {
        self._is_seeded = true;
        self._seed = q;
    }
}

impl<D: Distance + Base> AnnoyIndex<D> {
    pub fn new(f: usize, d: D) -> AnnoyIndex<D> {
        return AnnoyIndex {
            _verbose: false,
            _built: false,
            _f: f,
            _seed: 0,
            _K: (f as i32) + 2,
            distance: d,
            ..Default::default()
        };
    }

    pub fn get_f(&self) -> usize {
        self._f
    }

    pub fn get_node_mut(&mut self, i: i32) -> &mut Node {
        if self.nodes.len() <= i as usize {
            self.extent_nodes(i as usize);
        }
        return &mut self.nodes[i as usize];
    }

    pub fn get_node(&self, i: i32) -> Option<&Node> {
        if self.nodes.len() < i as usize {
            return None;
        }
        if self.nodes[i as usize].is_empty() {
            return None;
        }
        return Some(&self.nodes[i as usize]);
    }

    fn extent_nodes(&mut self, i: usize) {
        let diff = i - self.nodes.len() + 1;
        if diff > 0 {
            for i in 0..diff {
                self.nodes.push(Node::new());
            }
        }
    }

    // q => tree count
    fn thread_build(&mut self, q: i32) {
        random::set_seed(self._seed);

        let mut thread_root: Vec<i32> = Vec::new();

        loop {
            if q == -1 {
                self.thread_lock_nodes();
                if self._n_nodes >= 2 * self._n_items {
                    self.thread_unlock_nodes();
                    break;
                }
                self.thread_unlock_nodes();
            } else {
                if thread_root.len() >= (q as usize) {
                    break;
                }
            }

            let mut indices: Vec<i32> = Vec::new();
            self.thread_lock_nodes();
            for i in 0..self._n_nodes {
                match self.get_node(i) {
                    Some(node) => {
                        println!("hello {:?} {:?} ", i, self.nodes.len());

                        if node.n_descendants >= 1 {
                            indices.push(i as i32);
                        }
                    }
                    None => continue, // TODO: log
                }
            }
            self.thread_unlock_nodes();

            match self.make_tree(&indices, true) {
                Ok(tree) => thread_root.push(tree),
                Err(e) => continue,
            }
        }

        // thread lock
        self._roots.extend_from_slice(&thread_root);
    }

    fn thread_lock_nodes(&self) {}

    fn thread_unlock_nodes(&self) {}

    fn make_tree(&mut self, indices: &[i32], is_root: bool) -> Result<i32, &'static str> {
        if indices.len() == 0 {
            return Err("empty indices");
        }
        if indices.len() == 1 && !is_root {
            return Ok(indices[0]);
        }

        if (indices.len() as i32) <= self._K
            && (!is_root || self._n_items <= self._K || indices.len() == 1)
        {
            self._n_nodes += 1;
            let item = self._n_nodes;
            let mut n = self.get_node_mut(item);
            if n.is_empty() {
                n.init();
            }
            n.n_descendants = if is_root { item } else { indices.len() as i32 };
            for i in 0..indices.len() {
                println!("{:?} {:?}",i , indices.len());
                if n.children.len() == i {
                    n.children.push(indices[i].clone());
                }else {
                    n.children[i] = indices[i].clone();
                }
                
            }
            return Ok(item);
        }

        let mut children: Vec<Node> = Vec::new();
        for i in 0..indices.len() {
            let j = indices[i];
            match self.get_node(j) {
                None => continue,
                Some(node) => {
                    children.push(node.clone());
                }
            }
        }

        let mut m = Node::new();
        let mut children_indices: [Vec<i32>; 2] = [Vec::new(), Vec::new()];

        const attempt: usize = 3;
        for i in 0..attempt {
            children_indices[0].clear();
            children_indices[1].clear();
            self.distance.create_split(children.as_slice(), &mut m);

            for i in 0..indices.len() {
                let j = indices[i];
                match self.get_node(i as i32) {
                    Some(node) => {
                        let side = self.distance.side(&m, &node.v);
                        children_indices[(side as usize)].push(j);
                    }
                    None => continue,
                }
            }

            if self.split_imbalanced(&children_indices[0], &children_indices[1]) < 0.95 {
                break;
            }
        }

        while self.split_imbalanced(&children_indices[0], &children_indices[1]) > 0.95 {
            children_indices[0].clear();
            children_indices[1].clear();

            let is_initial = m.v.len() == 0;
            for z in 0..self._f {
                if is_initial {
                    m.v.push(0.0);
                } else {
                    m.v[z] = 0.0;
                }
            }

            for i in 0..indices.len() {
                let j = indices[i];
                children_indices[random::flip() as usize].push(j);
            }
        }

        let flip = (children_indices[0].len() > children_indices[1].len()) as bool;

        m.n_descendants = if is_root {
            self._n_items
        } else {
            indices.len() as i32
        };

        for side in 0..2 {
            match self.make_tree(&children_indices[side ^ (flip as usize)], false) {
                Ok(tree) => m.children[side ^ (flip as usize)] = tree,
                Err(e) => continue,
            }
        }

        self.nodes.push(m);
        self._n_nodes += 1;
        // let mut mn = self.get_node_mut(self._n_nodes);
        // Node::copy(&mut mn, &m);

        return Ok((self.nodes.len() - 1) as i32);
    }

    fn split_imbalanced(&self, left_indices: &[i32], right_indices: &[i32]) -> f64 {
        let ls = left_indices.len() as f64;
        let rs = right_indices.len() as f64;
        let f = ls / (ls + rs + 1e-9);
        if f > (1.0 - f) {
            return f;
        } else {
            return 1.0 - f;
        }
    }

    pub fn get_all_nns(
        &self,
        vectors: &[f64],
        n: usize,
        mut search_k: i32,
    ) -> Result<(Vec<i32>, Vec<f64>), &'static str> {
        let mut v_node = Node::new();
        self.distance.zero_value(&mut v_node);
        v_node.v = vectors.to_vec();
        self.distance.init_node(&mut v_node);

        if self._roots.len() == 0 {
            return Err("empty tree");
        }
        let node = Node::new();

        if search_k == -1 {
            search_k = (n * self._roots.len()) as i32;
        }

        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::new();
        for i in 0..self._roots.len() {
            heap.push(Neighbor {
                distance: self.distance.pq_initial_value(),
                idx: self._roots[i],
            });
        }

        let mut nns: Vec<i32> = Vec::new();
        while nns.len() < (search_k as usize) && !(heap.is_empty()) {
            let top = heap.peek().unwrap();
            let d = top.distance;
            let i = top.idx;
            println!("hello {:?} {:?}", self.nodes.len(), top);

            let nd = self.get_node(i).unwrap();
            heap.pop();

            if nd.n_descendants == 1 && i < self._n_items {
                nns.push(i.clone());
            } else if nd.n_descendants <= self._K {
                nns.extend_from_slice(&nd.children);
            } else {
                let margin = self.distance.margin(&nd, vectors)?;
                heap.push(Neighbor {
                    distance: self.distance.pq_distance(d, margin, 1),
                    idx: nd.children[1],
                });
                heap.push(Neighbor {
                    distance: self.distance.pq_distance(d, margin, 0),
                    idx: nd.children[0],
                });
            }
        }

        nns.sort();
        let mut nns_dist: Vec<Neighbor> = Vec::new();
        let mut last = -1;
        for i in 0..nns.len() {
            let j = nns[i];
            if j == last {
                continue;
            }
            last = j;
            let node = self.get_node(j).unwrap();
            if node.n_descendants == 1 {
                nns_dist.push(Neighbor {
                    distance: self.distance.distance(&v_node, &node)?,
                    idx: j,
                })
            }
        }

        let m = nns_dist.len();
        let p = if n < m { n } else { m };

        nns.sort();

        let mut result = Vec::new();
        let mut distance_result = Vec::new();

        for i in 0..p {
            result.push(nns_dist[i].idx);
            distance_result.push(nns_dist[i].distance);
        }

        println!("{:?}", self._roots);

        for i in 0..self.nodes.len() {
            println!("{:?} {:?}", i, self.nodes[i]);
        }

        return Ok((result, distance_result));
    }
}
