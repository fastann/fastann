use crate::annoy::random;
use crate::core::metrics;
use crate::core::neighbor;
use crate::core::node;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::marker::PhantomData;

const ITERATION_STEPS: usize = 200;

// TODO: leaf as a trait with getter setter function
#[derive(Default, Clone, Debug)]
pub struct Leaf<E: node::FloatElement> {
    n_descendants: i32, // tot n_descendants
    children: Vec<i32>, // left and right and if it's a leaf leaf, children would be very large (depend on _K)
    node: Box<node::Node<E>>,

    // biz field
    norm: E,
    has_init: bool,
}

impl<E: node::FloatElement> Leaf<E> {
    fn new() -> Leaf<E> {
        Leaf {
            children: vec![0, 0],
            ..Default::default()
        }
    }

    fn new_with_vectors(_v: &[E]) -> Leaf<E> {
        Leaf {
            children: vec![0, 0],
            node: Box::new(node::Node::new(_v)),
            ..Default::default()
        }
    }

    fn is_empty(&self) -> bool {
        return self.has_init;
    }

    fn init(&mut self) {
        self.children = vec![0, 0];
    }

    fn copy(dst: &mut Leaf<E>, src: &Leaf<E>) {
        dst.n_descendants = src.n_descendants.clone();
        dst.children = src.children.clone();
        dst.node = src.node.clone();
        dst.norm = src.norm.clone();
    }

    pub fn get_literal(&self) -> String {
        format!(
            "{{ \"n_descendants\": {:?}, \"children\": {:?}, \"norm\": {:?}, \"has_init\": {:?} }}",
            self.n_descendants, self.children, self.norm, self.has_init
        )
    }
}

pub fn two_means<D: Distance<E> + Base<E>, E: node::FloatElement>(
    leaves: &[Leaf<E>],
    use_cosine: bool,
    distance: &D,
) -> Result<(Leaf<E>, Leaf<E>), &'static str> {
    if leaves.len() < 2 {
        return Err("empty leaves");
    }
    let count = leaves.len();
    let i = random::index(count);
    let mut j = random::index(count - 1);

    if j >= i {
        j += 1;
    }

    let mut p = distance.copy_leaf(&leaves[i]);
    let mut q = distance.copy_leaf(&leaves[j]);

    if use_cosine {
        distance.normalize(&mut p);
        distance.normalize(&mut q);
    }
    distance.init_leaf(&mut p);
    distance.init_leaf(&mut q);

    let mut ic: E = E::one();
    let mut jc: E = E::one();

    for _z in 0..ITERATION_STEPS {
        let k = random::index(count);
        let di = ic * distance.distance(&p, &leaves[k])?;
        let dj = jc * distance.distance(&q, &leaves[k])?;
        let norm = if use_cosine {
            metrics::get_norm(&leaves[k].node.vectors())
        } else {
            E::one()
        };

        if !(norm > E::float_zero()) {
            continue;
        }

        if di < dj {
            for l in 0..q.node.len() {
                p.node.mut_vectors()[l] = (p.node.vectors()[l] * ic
                    + leaves[k].node.vectors()[l] / norm)
                    / (ic + E::one());
            }
            distance.init_leaf(&mut p);
            ic += E::float_one();
        } else if dj < di {
            for l in 0..q.node.len() {
                q.node.mut_vectors()[l] = (q.node.vectors()[l] * ic
                    + leaves[k].node.vectors()[l] / norm)
                    / (ic + E::float_one());
            }
            distance.init_leaf(&mut q);
            jc += E::float_one();
        }
    }

    return Ok((p, q));
}

pub trait Base<E: node::FloatElement>: Default {
    // TODO:
    fn preprocess(&self, leaves: &[Leaf<E>]) {}

    fn zero_value(&self, src: &mut Leaf<E>) {}
    fn copy_leaf(&self, src: &Leaf<E>) -> Leaf<E> {
        return Leaf {
            n_descendants: src.n_descendants.clone(),
            node: src.node.clone(),
            children: src.children.clone(),
            ..Default::default()
        };
    }
    fn normalize(&self, leaf: &mut Leaf<E>) {
        let norm = metrics::get_norm(&leaf.node.vectors());
        if norm > E::float_zero() {
            for i in 0..leaf.node.len() {
                leaf.node.mut_vectors()[i] /= norm;
            }
        }
    }
}

pub trait Distance<E: node::FloatElement> {
    fn init_leaf(&self, leaf: &mut Leaf<E>) {}
    fn distance(&self, src: &Leaf<E>, dst: &Leaf<E>) -> Result<E, &'static str>;
    fn create_split(&self, leaves: &[Leaf<E>], n: &mut Leaf<E>) -> Result<(), &'static str>;
    fn pq_initial_value(&self) -> E {
        return E::max_value();
    }
    fn side(&self, src: &Leaf<E>, dst: &[E]) -> bool {
        return false;
    }

    fn margin(&self, src: &Leaf<E>, dst: &[E]) -> Result<E, &'static str> {
        return Ok(E::float_zero());
    }

    fn pq_distance(&self, distance: E, mut margin: E, child_nr: usize) -> E {
        return E::float_zero();
    }
}

#[derive(Default, Clone, Debug, Copy)]
pub struct Angular<E: node::FloatElement> {
    phantom: PhantomData<E>, // empty size variable
}

impl<E: node::FloatElement> Angular<E> {
    pub fn new() -> Self {
        Angular {
            phantom: PhantomData,
        }
    }
}

impl<E: node::FloatElement> Base<E> for Angular<E> {
    fn copy_leaf(&self, src: &Leaf<E>) -> Leaf<E> {
        return src.clone();
    }
}

impl<E: node::FloatElement> Distance<E> for Angular<E> {
    // want to metricsulate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    fn distance(&self, src: &Leaf<E>, dst: &Leaf<E>) -> Result<E, &'static str> {
        let left = if src.norm != E::float_zero() {
            src.norm
        } else {
            metrics::dot(&src.node.vectors(), &src.node.vectors())?
        };
        let right = if dst.norm != E::float_zero() {
            dst.norm
        } else {
            metrics::dot(&dst.node.vectors(), &dst.node.vectors())?
        };
        let dot_val = metrics::dot(&src.node.vectors(), &dst.node.vectors())?;
        let inner_val = right * left;
        let two = E::from_f32(2.0).unwrap();
        if inner_val > E::float_zero() {
            return Result::Ok(two - two * dot_val / inner_val.sqrt());
        } else {
            return Result::Ok(two);
        }
    }

    fn margin(&self, src: &Leaf<E>, dst: &[E]) -> Result<E, &'static str> {
        return metrics::dot(&src.node.vectors(), &dst);
    }

    fn side(&self, src: &Leaf<E>, dst: &[E]) -> bool {
        match self.margin(&src, &dst) {
            Ok(x) => {
                return x > E::float_zero();
            }
            Err(e) => {
                return random::flip();
            }
        }
    }

    // use euclidean distance
    fn create_split(&self, leaves: &[Leaf<E>], n: &mut Leaf<E>) -> Result<(), &'static str> {
        let (p, q) = two_means(&leaves, true, self)?;

        if n.node.len() != 0 && n.node.len() != p.node.len() {
            return Err("empty leaf input");
        }

        let is_initial = if n.node.len() == 0 { true } else { false };
        for i in 0..p.node.len() {
            if is_initial {
                n.node
                    .mut_vectors()
                    .push(p.node.vectors()[i] - q.node.vectors()[i]);
            } else {
                n.node.mut_vectors()[i] = p.node.vectors()[i] - q.node.vectors()[i];
            }
        }
        self.normalize(n);
        return Ok(());
    }

    fn init_leaf(&self, leaf: &mut Leaf<E>) {
        match metrics::dot(&leaf.node.vectors(), &leaf.node.vectors()) {
            Ok(dot) => {
                leaf.norm = dot;
            }
            Err(e) => return, // do nothing
        }
    }

    fn pq_distance(&self, distance: E, mut margin: E, child_nr: usize) -> E {
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

impl<E: node::FloatElement> Angular<E> {
    fn normalize_distances(&self, distance: E) -> E {
        if distance > E::float_zero() {
            return distance.sqrt();
        }
        return E::float_zero();
    }

    fn pq_initial_value(&self) -> E {
        return E::max_value();
    }

    fn name(&self) -> &'static str {
        "angular"
    }
}

#[derive(Default, Clone, Debug)]
pub struct DotProduct<E: node::FloatElement> {
    angular: Angular<E>,
}

impl<E: node::FloatElement> Distance<E> for DotProduct<E> {
    fn distance(&self, src: &Leaf<E>, dst: &Leaf<E>) -> Result<E, &'static str> {
        return Ok(-metrics::dot(&src.node.vectors(), &dst.node.vectors())?);
    }

    fn create_split(&self, leaves: &[Leaf<E>], n: &mut Leaf<E>) -> Result<(), &'static str> {
        return Ok(self.angular.create_split(&leaves, n)?);
    }

    fn init_leaf(&self, leaf: &mut Leaf<E>) {}
}

impl<E: node::FloatElement> Base<E> for DotProduct<E> {}

// TODO: implement
impl<E: node::FloatElement> DotProduct<E> {}

pub trait AnnoyIndexer<E: node::FloatElement, D: Distance<E> + Base<E>> {
    fn add_item(&mut self, item: i32, w: &[E], d: D) -> Result<(), &'static str>;
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
    fn get_distance(&self, i: i32, j: i32) -> Result<E, &'static str> {
        return Ok(E::float_zero());
    }
    fn get_nns_by_item(
        &self,
        idx: i32,
        n: usize,
        search_k: i32,
    ) -> Result<(Vec<i32>, Vec<E>), &'static str> {
        return Err("please implement the method");
    }
    fn get_nns_by_vector(
        &self,
        f: &[E],
        n: usize,
        search_k: i32,
    ) -> Result<(Vec<i32>, Vec<E>), &'static str> {
        return Err("please implement the method");
    }
    fn get_tot_items_cnt(&self) -> i32 {
        return 0;
    }
    fn get_n_tree(&self) -> i32 {
        return 0;
    }
    fn verbose(&self, v: bool) {}
    fn get_item(&self, item: E) -> Result<E, &'static str> {
        return Ok(E::float_zero());
    }
    fn set_seed(&mut self, q: i32) {}
    // fn on_disk_build
}

#[derive(Default, Debug)]
pub struct AnnoyIndex<E: node::FloatElement, D: Distance<E> + Base<E>> {
    _f: usize, // dimension
    // _s: i32,       // leaf size
    _tot_items_cnt: i32, // add items count, means the physically the item count, _tot_items_cnt == leaves.size()
    // _leaves;
    _tot_leaves_cnt: i32, // leaves count, whole tree leaves count
    // _leaves_size: i32, // in source code, this means the memory which has been allocated, and we can use leaf's size to get data
    _roots: Vec<i32>, // dummy root's children
    _K: i32,          // max number of n_descendants to fit into leaf
    _is_seeded: bool,
    _seed: i32,
    _loaded: bool,
    _verbose: bool,
    _fd: i32,
    _on_disk: bool,
    _built: bool,
    pub leaves: Vec<Leaf<E>>,

    distance: D,
}

impl<E: node::FloatElement, D: Distance<E> + Base<E>> AnnoyIndexer<E, D> for AnnoyIndex<E, D> {
    fn add_item(&mut self, item: i32, w: &[E], d: D) -> Result<(), &'static str> {
        // TODO: remove
        if w.len() != self._f {
            return Err("dimension is different");
        }

        // TODO:
        if self._loaded {
            return Err("you can't add an item to a loaded index");
        }

        let mut nn = Leaf::new();

        d.zero_value(&mut nn);

        nn.children[0] = 0; // TODO: as const value
        nn.children[1] = 0;
        nn.n_descendants = 1; // only the leaf itself, so the n_descendants include it self

        nn.node.set_vectors(w);

        d.init_leaf(&mut nn);

        // no update method
        self._tot_items_cnt += 1;

        self.leaves.push(nn);

        return Ok(());
    }

    fn build(&mut self, q: i32) -> Result<(), &'static str> {
        if self._built {
            return Err("has built");
        }

        self.distance.preprocess(&self.leaves);

        self._tot_leaves_cnt = self._tot_items_cnt;
        self.thread_build(q);
        self._built = true;
        return Ok(());
    }

    fn unbuild(&mut self) -> Result<(), &'static str> {
        self._roots.clear();
        self._tot_leaves_cnt = self._tot_items_cnt;
        self._built = false;
        return Ok(());
    }
    fn get_distance(&self, i: i32, j: i32) -> Result<E, &'static str> {
        let ni = self.get_leaf(i).unwrap();
        let nj = self.get_leaf(j).unwrap();
        return Ok(self.distance.distance(&ni, &nj)?);
    }

    fn get_nns_by_item(
        &self,
        idx: i32,
        n: usize,
        search_k: i32,
    ) -> Result<(Vec<i32>, Vec<E>), &'static str> {
        match self.get_leaf(idx) {
            Some(leaf) => self.get_all_nns(&leaf.node.vectors(), n, search_k),
            None => return Err("invalid idx"),
        }
    }

    fn get_nns_by_vector(
        &self,
        f: &[E],
        n: usize,
        search_k: i32,
    ) -> Result<(Vec<i32>, Vec<E>), &'static str> {
        self.get_all_nns(&f, n, search_k)
    }

    fn get_tot_items_cnt(&self) -> i32 {
        return self._tot_items_cnt;
    }
    fn get_n_tree(&self) -> i32 {
        return self._roots.len() as i32;
    }

    fn set_seed(&mut self, q: i32) {
        self._is_seeded = true;
        self._seed = q;
    }
}

impl<E: node::FloatElement, D: Distance<E> + Base<E>> AnnoyIndex<E, D> {
    pub fn new(f: usize, d: D) -> AnnoyIndex<E, D> {
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

    pub fn get_K(&self) -> i32 {
        self._K
    }

    pub fn get_leaf_mut(&mut self, i: i32) -> &mut Leaf<E> {
        if self.leaves.len() <= i as usize {
            self.extent_leaves(i as usize);
        }
        return &mut self.leaves[i as usize];
    }

    pub fn get_leaf(&self, i: i32) -> Option<&Leaf<E>> {
        if self.leaves.len() < i as usize {
            return None;
        }
        if self.leaves[i as usize].is_empty() {
            return None;
        }
        return Some(&self.leaves[i as usize]);
    }

    fn extent_leaves(&mut self, i: usize) {
        let diff = i - self.leaves.len() + 1;
        if diff > 0 {
            for i in 0..diff {
                self.leaves.push(Leaf::new());
            }
        }
    }

    // q => tree count
    fn thread_build(&mut self, q: i32) {
        random::set_seed(self._seed);

        let mut thread_root: Vec<i32> = Vec::new();

        loop {
            if q == -1 {
                self.thread_lock_leaves();
                if self._tot_leaves_cnt >= 2 * self._tot_items_cnt {
                    self.thread_unlock_leaves();
                    break;
                }
                self.thread_unlock_leaves();
            } else {
                if thread_root.len() >= (q as usize) {
                    break;
                }
            }

            let mut indices: Vec<i32> = Vec::new();
            self.thread_lock_leaves();
            for i in 0..self._tot_items_cnt {
                match self.get_leaf(i) {
                    Some(leaf) => {
                        if leaf.n_descendants >= 1 {
                            indices.push(i as i32);
                        }
                    }
                    None => continue, // TODO: log
                }
            }
            self.thread_unlock_leaves();

            match self.make_tree(&indices, true) {
                Ok(tree) => {
                    thread_root.push(tree);
                }
                Err(e) => {
                    continue;
                } // TODO log
            }
        }

        // thread lock
        self._roots.extend_from_slice(&thread_root);
    }

    fn thread_lock_leaves(&self) {}

    fn thread_unlock_leaves(&self) {}

    fn make_tree(&mut self, indices: &[i32], is_root: bool) -> Result<i32, &'static str> {
        if indices.len() == 0 {
            return Err("empty indices");
        }
        if indices.len() == 1 && !is_root {
            return Ok(indices[0]);
        }

        // the batch is a leaf cluster, make a parent node
        if (indices.len() as i32) <= self._K
            && (!is_root || self._tot_items_cnt <= self._K || indices.len() == 1)
        {
            self._tot_leaves_cnt += 1;
            let item = self._tot_leaves_cnt;
            let mut n = self.get_leaf_mut(item);
            if n.is_empty() {
                n.init();
            }
            n.n_descendants = if is_root { item } else { indices.len() as i32 };
            for i in 0..indices.len() {
                if n.children.len() == i {
                    n.children.push(indices[i].clone());
                } else {
                    n.children[i] = indices[i].clone();
                }
            }
            return Ok(item);
        }

        let mut children: Vec<Leaf<E>> = Vec::new();
        for i in 0..indices.len() {
            let j = indices[i];
            match self.get_leaf(j) {
                None => continue,
                Some(leaf) => {
                    children.push(leaf.clone());
                }
            }
        }

        let mut new_parent_leaf = Leaf::new();
        let mut children_indices: [Vec<i32>; 2] = [Vec::new(), Vec::new()];

        const attempt: usize = 3;
        for i in 0..attempt {
            children_indices[0].clear();
            children_indices[1].clear();
            self.distance
                .create_split(children.as_slice(), &mut new_parent_leaf);

            for i in 0..indices.len() {
                let j = indices[i];
                match self.get_leaf(i as i32) {
                    Some(leaf) => {
                        let side = self.distance.side(&new_parent_leaf, &leaf.node.vectors());
                        children_indices[(side as usize)].push(j);
                    }
                    None => continue,
                }
            }

            if self.split_imbalanced(&children_indices[0], &children_indices[1])
                < E::from_f32(0.95).unwrap()
            {
                break;
            }
        }

        while self.split_imbalanced(&children_indices[0], &children_indices[1])
            > E::from_f32(0.98).unwrap()
        {
            children_indices[0].clear();
            children_indices[1].clear();

            let is_initial = new_parent_leaf.node.len() == 0;
            for z in 0..self._f {
                if is_initial {
                    new_parent_leaf.node.push(&E::float_zero()); // TODO: make it const value
                } else {
                    new_parent_leaf.node.mut_vectors()[z] = E::float_zero();
                }
            }

            for i in 0..indices.len() {
                let j = indices[i];
                children_indices[random::flip() as usize].push(j);
            }
        }

        let flip = (children_indices[0].len() > children_indices[1].len()) as bool;

        new_parent_leaf.n_descendants = if is_root {
            self._tot_items_cnt
        } else {
            indices.len() as i32
        };

        for side in 0..2 {
            match self.make_tree(&children_indices[side ^ (flip as usize)], false) {
                Ok(tree) => {
                    new_parent_leaf.children[side ^ (flip as usize)] = tree;
                }
                Err(e) => {
                    // TODO: log
                    continue;
                }
            }
        }
        self._tot_leaves_cnt += 1;
        self.leaves.push(new_parent_leaf);

        return Ok((self._tot_leaves_cnt) as i32);
    }

    fn split_imbalanced(&self, left_indices: &[i32], right_indices: &[i32]) -> E {
        let ls = E::from_usize(left_indices.len()).unwrap();
        let rs = E::from_usize(right_indices.len()).unwrap();
        let f = ls / (ls + rs + E::from_f32(1e-9).unwrap());
        if f > (E::float_one() - f) {
            return f;
        } else {
            return E::float_one() - f;
        }
    }

    pub fn get_all_nns(
        &self,
        vectors: &[E],
        n: usize,
        mut search_k: i32,
    ) -> Result<(Vec<i32>, Vec<E>), &'static str> {
        let mut v_leaf = Leaf::new();
        self.distance.zero_value(&mut v_leaf);
        v_leaf.node.set_vectors(&vectors.to_vec());
        self.distance.init_leaf(&mut v_leaf);

        if self._roots.len() == 0 {
            return Err("empty tree");
        }
        let leaf: Leaf<E> = Leaf::new();

        if search_k == -1 {
            search_k = (n * self._roots.len()) as i32;
        }

        let mut heap: BinaryHeap<neighbor::Neighbor<E>> = BinaryHeap::new();
        for i in 0..self._roots.len() {
            heap.push(neighbor::Neighbor {
                _distance: self.distance.pq_initial_value(),
                _idx: self._roots[i] as usize,
            });
        }

        let mut nns: Vec<i32> = Vec::new();
        while nns.len() < (search_k as usize) && !(heap.is_empty()) {
            let top = heap.peek().unwrap();
            let d = top._distance;
            let i = top._idx;

            let nd = self.get_leaf(i as i32).unwrap();
            heap.pop();

            if nd.n_descendants == 1 && (i as i32) < self._tot_items_cnt {
                nns.push(i.clone() as i32);
            } else if nd.n_descendants <= self._K {
                nns.extend_from_slice(&nd.children);
            } else {
                let margin = self.distance.margin(&nd, vectors)?;
                heap.push(neighbor::Neighbor {
                    _distance: self.distance.pq_distance(d, margin, 1),
                    _idx: nd.children[1] as usize,
                });
                heap.push(neighbor::Neighbor {
                    _distance: self.distance.pq_distance(d, margin, 0),
                    _idx: nd.children[0] as usize,
                });
            }
        }

        nns.sort();
        let mut nns_dist: Vec<neighbor::Neighbor<E>> = Vec::new();
        let mut last = -1;
        for i in 0..nns.len() {
            let j = nns[i];
            if j == last {
                continue;
            }
            last = j;
            let leaf = self.get_leaf(j).unwrap();
            if leaf.n_descendants == 1 {
                nns_dist.push(neighbor::Neighbor {
                    _distance: self.distance.distance(&v_leaf, &leaf)?,
                    _idx: j as usize,
                })
            }
        }

        let m = nns_dist.len();
        let p = if n < m { n } else { m };

        nns.sort();

        let mut result: Vec<i32> = Vec::new();
        let mut distance_result = Vec::new();

        for i in 0..p {
            result.push(nns_dist[i]._idx as i32);
            distance_result.push(nns_dist[i]._distance);
        }

        return Ok((result, distance_result));
    }

    pub fn show_trees(&self) {
        let mut v = self._roots.clone();

        while !v.is_empty() {
            let i = v.pop().unwrap();
            println!("get item {}", i);
            let item = self.get_leaf(i).unwrap();
            if !(item.children[0] == 0 && item.children[1] == 0) {
                v.extend(&item.children);
            }
            println!(
                "item {} children {:?}, vectors {:?}",
                i,
                item.children,
                item.node.vectors()
            );
        }
    }
}
