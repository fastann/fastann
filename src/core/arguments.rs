use crate::core::node;
use hashbrown::HashMap;

// TODO: make this optional
pub struct Arguments {
    float_args: HashMap<String, f32>,
    int_args: HashMap<String, i32>,
    str_args: HashMap<String, String>,
}

impl Arguments {
    pub fn new() -> Self {
        Arguments {
            float_args: HashMap::new(),
            int_args: HashMap::new(),
            str_args: HashMap::new(),
        }
    }

    pub fn fget(&self, key: &str) -> Option<f32> {
        match self.float_args.get(key) {
            Some(val) => Some(val.clone()),
            None => None,
        }
    }

    pub fn iget(&self, key: &str) -> Option<i32> {
        match self.int_args.get(key) {
            Some(val) => Some(val.clone()),
            None => None,
        }
    }

    pub fn sget(&self, key: &str) -> Option<String> {
        match self.str_args.get(key) {
            Some(val) => Some(val.clone()),
            None => None,
        }
    }

    pub fn fset(&mut self, key: &str, value: f32) -> &mut Arguments {
        self.float_args.insert(key.to_string(), value.clone());
        self
    }

    pub fn iset(&mut self, key: &str, value: i32) -> &mut Arguments {
        self.int_args.insert(key.to_string(), value.clone());
        self
    }

    pub fn sset(&mut self, key: &str, value: &str) -> &mut Arguments {
        self.str_args.insert(key.to_string(), value.to_string());
        self
    }
}
