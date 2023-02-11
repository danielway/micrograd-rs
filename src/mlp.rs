use crate::{layer::Layer, value::Value};

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(input_count: usize, output_counts: Vec<usize>) -> MLP {
        let output_counts_len = output_counts.len();
        let layer_sizes: Vec<usize> = [input_count].into_iter().chain(output_counts).collect();

        MLP {
            layers: (0..output_counts_len)
                .map(|i| Layer::new(layer_sizes[i], layer_sizes[i + 1]))
                .collect(),
        }
    }

    pub fn forward(&self, mut xs: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            xs = layer.forward(&xs);
        }
        xs
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
