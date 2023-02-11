use crate::{neuron::Neuron, value::Value};

#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(input_count: usize, output_count: usize) -> Layer {
        Layer {
            neurons: (0..output_count)
                .map(|_| Neuron::new(input_count))
                .collect(),
        }
    }

    pub fn forward(&self, xs: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(xs)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}
