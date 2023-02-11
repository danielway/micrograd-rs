//! This is a basic example of performing back-propagation on a two-input/weight graph.

use micrograd_rs::Value;

fn main() {
    let x1 = Value::from(2.0).with_label("x1");
    let x1_clone = x1.clone();
    let x2 = Value::from(0.0).with_label("x2");

    let w1 = Value::from(-3.0).with_label("w1");
    let w2 = Value::from(1.0).with_label("w2");

    let b = Value::from(6.8813735870195432).with_label("b");

    let x1w1 = (x1 * w1).with_label("x1w1");
    let x2w2 = (x2 * w2).with_label("x2w2");

    let x1w1x2w2 = (x1w1 + x2w2).with_label("x1w1x2w2");

    let n = (x1w1x2w2 + b).with_label("n");
    let o = n.tanh().with_label("o");

    o.backward();

    assert_eq!(0.7071, round_to(o.data(), 4.0));
    assert_eq!(-1.5, round_to(x1_clone.gradient(), 3.0));
    println!("{:?}", o);
}

fn round_to(value: f64, digits: f64) -> f64 {
    let ten: f64 = 10.0;
    (ten.powf(digits) * value).round() / ten.powf(digits)
}
