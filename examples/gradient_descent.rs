//! This is a basic example of performing gradient descent with a neural network using micrograd-rs.

use micrograd_rs::{Value, MLP};

fn main() {
    let mlp = MLP::new(3, vec![4, 4, 1]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    let ys = vec![1.0, -1.0, -1.0, 1.0];

    for _ in 0..100 {
        // Forward pass
        let ypred: Vec<Value> = xs
            .iter()
            .map(|x| mlp.forward(x.iter().map(|x| Value::from(*x)).collect())[0].clone())
            .collect();
        let ypred_floats: Vec<f64> = ypred.iter().map(|v| v.data()).collect();

        // Loss function
        let ygt = ys.iter().map(|y| Value::from(*y));
        let loss: Value = ypred
            .into_iter()
            .zip(ygt)
            .map(|(yp, yg)| (yp - yg).pow(&Value::from(2.0)))
            .sum();

        println!("Loss: {} Predictions: {:?}", loss.data(), ypred_floats);

        // Backward pass
        mlp.parameters().iter().for_each(|p| p.clear_gradient());
        loss.backward();

        // Adjustment
        mlp.parameters().iter().for_each(|p| p.adjust(-0.05));
    }
}
