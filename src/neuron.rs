use crate::value::Value;

use rand::{thread_rng, Rng};

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(input_count: usize) -> Neuron {
        let mut rng = thread_rng();
        let mut rand_value_fn = || {
            let data = rng.gen_range(-1.0..1.0);
            Value::from(data)
        };

        let mut weights = Vec::new();
        for _ in 0..input_count {
            weights.push(rand_value_fn());
        }

        Neuron {
            weights,
            bias: rand_value_fn().with_label("b"),
        }
    }

    pub fn forward(&self, xs: &Vec<Value>) -> Value {
        let products = std::iter::zip(&self.weights, xs)
            .map(|(a, b)| a * b)
            .collect::<Vec<Value>>();

        let sum = self.bias.clone() + products.into_iter().reduce(|acc, prd| acc + prd).unwrap();
        sum.tanh()
    }

    pub fn parameters(&self) -> Vec<Value> {
        [self.bias.clone()]
            .into_iter()
            .chain(self.weights.clone())
            .collect()
    }
}
