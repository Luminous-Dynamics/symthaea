// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Production-grade Generic EML Symbolic Regression with Parallel Search

use super::autodiff::{
    ad_begin_c64, ad_begin_f64, ad_end, ad_gradient_c64, ad_gradient_f64, GenericVar, Scalar,
};
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use std::marker::PhantomData;

fn differentiable_softmax<T: Scalar>(
    logits: &[GenericVar<T>],
    temperature: f64,
) -> Vec<GenericVar<T>> {
    if logits.is_empty() {
        return vec![];
    }
    let exps: Vec<GenericVar<T>> = logits
        .iter()
        .map(|v| v.div(GenericVar::constant(T::from_f64(temperature))).exp())
        .collect();
    let sum = exps
        .iter()
        .fold(GenericVar::constant(T::zero()), |acc, v| acc.add(*v));
    let safe_sum = sum.add(GenericVar::constant(T::from_f64(1e-15)));
    exps.iter().map(|v| v.div(safe_sum)).collect()
}

#[derive(Debug, Clone)]
pub struct EmlMasterNode<T: Scalar> {
    pub left_logits: Vec<f64>,
    pub right_logits: Vec<f64>,
    pub children: Option<(Box<EmlMasterNode<T>>, Box<EmlMasterNode<T>>)>,
    _marker: PhantomData<T>,
}

impl<T: Scalar> EmlMasterNode<T> {
    pub fn new(depth: usize, num_vars: usize, identity_bias: bool) -> Self {
        let mut rng = rand::thread_rng();
        let size = num_vars + 2;
        let mut left_logits = (0..size)
            .map(|_| rng.gen_range(-0.2..0.2))
            .collect::<Vec<_>>();
        let mut right_logits = (0..size)
            .map(|_| rng.gen_range(-0.2..0.2))
            .collect::<Vec<_>>();
        if identity_bias {
            left_logits[size - 1] = 5.0;
            right_logits[0] = 5.0;
        }
        let children = if depth > 0 {
            Some((
                Box::new(EmlMasterNode::<T>::new(depth - 1, num_vars, identity_bias)),
                Box::new(EmlMasterNode::<T>::new(depth - 1, num_vars, identity_bias)),
            ))
        } else {
            None
        };
        Self {
            left_logits,
            right_logits,
            children,
            _marker: PhantomData,
        }
    }

    pub fn grow(&mut self, num_vars: usize) {
        if let Some((left, right)) = &mut self.children {
            left.grow(num_vars);
            right.grow(num_vars);
        } else {
            let mut left_child = EmlMasterNode::<T>::new(0, num_vars, false);
            left_child.seed_logits(1, 0, 5.0);
            let mut right_child = EmlMasterNode::<T>::new(0, num_vars, false);
            right_child.seed_logits(0, 0, 5.0);
            self.children = Some((Box::new(left_child), Box::new(right_child)));
            self.seed_logits(self.left_logits.len() - 1, 0, 5.0);
        }
    }

    pub fn seed_logits(&mut self, left_idx: usize, right_idx: usize, strength: f64) {
        for i in 0..self.left_logits.len() {
            self.left_logits[i] = if i == left_idx { strength } else { -strength };
        }
        for i in 0..self.right_logits.len() {
            self.right_logits[i] = if i == right_idx { strength } else { -strength };
        }
    }

    pub fn eval(
        &self,
        vars: &[GenericVar<T>],
        params: &[GenericVar<T>],
        param_idx: &mut usize,
        temp: f64,
    ) -> GenericVar<T> {
        let size = self.left_logits.len();
        let l_logits_vars = &params[*param_idx..*param_idx + size];
        *param_idx += size;
        let r_logits_vars = &params[*param_idx..*param_idx + size];
        *param_idx += size;
        let l_probs = differentiable_softmax(l_logits_vars, temp);
        let r_probs = differentiable_softmax(r_logits_vars, temp);
        let (f_l, f_r) = if let Some((left, right)) = &self.children {
            (
                left.eval(vars, params, param_idx, temp),
                right.eval(vars, params, param_idx, temp),
            )
        } else {
            (
                GenericVar::constant(T::one()),
                GenericVar::constant(T::one()),
            )
        };
        let mut l = l_probs[0].mul(GenericVar::constant(T::one()));
        for i in 0..vars.len() {
            l = l.add(l_probs[i + 1].mul(vars[i]));
        }
        l = l.add(l_probs[size - 1].mul(f_l));
        let mut r = r_probs[0].mul(GenericVar::constant(T::one()));
        for i in 0..vars.len() {
            r = r.add(r_probs[i + 1].mul(vars[i]));
        }
        r = r.add(r_probs[size - 1].mul(f_r));
        l.eml(r)
    }

    pub fn collect_weights(&self, weights: &mut Vec<f64>) {
        weights.extend_from_slice(&self.left_logits);
        weights.extend_from_slice(&self.right_logits);
        if let Some((left, right)) = &self.children {
            left.collect_weights(weights);
            right.collect_weights(weights);
        }
    }

    pub fn update_weights(&mut self, weights: &[f64], idx: &mut usize) {
        let size = self.left_logits.len();
        self.left_logits
            .copy_from_slice(&weights[*idx..*idx + size]);
        *idx += size;
        self.right_logits
            .copy_from_slice(&weights[*idx..*idx + size]);
        *idx += size;
        if let Some((left, right)) = &mut self.children {
            left.update_weights(weights, idx);
            right.update_weights(weights, idx);
        }
    }

    pub fn snap(&mut self) {
        let snap_group = |logits: &mut Vec<f64>| {
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            for i in 0..logits.len() {
                logits[i] = if i == max_idx { 15.0 } else { -15.0 };
            }
        };
        snap_group(&mut self.left_logits);
        snap_group(&mut self.right_logits);
        if let Some((left, right)) = &mut self.children {
            left.snap();
            right.snap();
        }
    }
}

pub struct EmlRegressor<T: Scalar + Send + Sync> {
    pub root: EmlMasterNode<T>,
    pub num_vars: usize,
    pub learning_rate: f64,
    pub entropy_weight: f64,
    _marker: PhantomData<T>,
}

impl<T: Scalar + Send + Sync> EmlRegressor<T> {
    pub fn new(depth: usize, num_vars: usize) -> Self {
        Self {
            root: EmlMasterNode::<T>::new(depth, num_vars, false),
            num_vars,
            learning_rate: 0.1,
            entropy_weight: 0.02,
            _marker: PhantomData,
        }
    }

    pub fn parallel_train(
        &mut self,
        dataset: &[(Vec<T>, T)],
        epochs: usize,
        seeds: usize,
        max_depth: usize,
    ) {
        let best_root = (0..seeds)
            .into_par_iter()
            .map(|_| {
                let mut reg = EmlRegressor::<T>::new(0, self.num_vars);
                reg.train_curriculum(dataset, max_depth, epochs);
                let mse = reg.calculate_mse(dataset);
                (mse, reg.root)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, root)| root)
            .unwrap();
        self.root = best_root;
    }

    pub fn train_curriculum(&mut self, dataset: &[(Vec<T>, T)], max_depth: usize, epochs: usize) {
        for _ in 0..=max_depth {
            self.train(dataset, epochs);
            if self.calculate_mse(dataset) < 1e-4 {
                break;
            }
            self.root.grow(self.num_vars);
        }
    }

    pub fn calculate_mse(&self, dataset: &[(Vec<T>, T)]) -> f64 {
        let mut total = 0.0;
        for (x, y) in dataset {
            total += (self.predict_inner(x) - *y).norm_sq();
        }
        total / dataset.len() as f64
    }

    pub fn train(&mut self, dataset: &[(Vec<T>, T)], epochs: usize) {
        let mut weights = Vec::new();
        self.root.collect_weights(&mut weights);
        let mut m = vec![0.0; weights.len()];
        let mut v = vec![0.0; weights.len()];
        let mut best_mse = f64::INFINITY;
        let mut plateau_count = 0;
        let mut current_lr = self.learning_rate;
        for epoch in 1..=epochs {
            let temp = 0.05 + 0.95 * (-6.0 * (epoch as f64 / epochs as f64)).exp();
            let mut total_grad = vec![0.0; weights.len()];
            for (x_vals, y_target) in dataset {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    ad_begin_f64();
                    let vars: Vec<GenericVar<f64>> = x_vals
                        .iter()
                        .map(|v| GenericVar::<f64>::new(v.to_f64()))
                        .collect();
                    let params: Vec<GenericVar<f64>> =
                        weights.iter().map(|&w| GenericVar::<f64>::new(w)).collect();
                    let mut idx = 0;
                    let root_f64: &EmlMasterNode<f64> = unsafe {
                        &*(&self.root as *const EmlMasterNode<T> as *const EmlMasterNode<f64>)
                    };
                    let prediction = root_f64.eval(&vars, &params, &mut idx, temp);
                    let loss = prediction
                        .sub(GenericVar::constant(y_target.to_f64()))
                        .mul(prediction.sub(GenericVar::constant(y_target.to_f64())));
                    let adjoints = ad_gradient_f64(loss);
                    for (i, p) in params.iter().enumerate() {
                        total_grad[i] += adjoints[p.index]
                            + if weights[i].abs() < 1.0 {
                                -self.entropy_weight * weights[i].signum()
                            } else {
                                0.0
                            };
                    }
                    ad_end();
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
                    ad_begin_c64();
                    let vars: Vec<GenericVar<Complex64>> = x_vals
                        .iter()
                        .map(|v| {
                            let c: Complex64 = unsafe {
                                *(&v as *const &T as *const *const Complex64 as *const Complex64)
                            };
                            GenericVar::<Complex64>::new(c)
                        })
                        .collect();
                    let params: Vec<GenericVar<Complex64>> = weights
                        .iter()
                        .map(|&w| GenericVar::<Complex64>::new(Complex64::new(w, 0.0)))
                        .collect();
                    let mut idx = 0;
                    let root_c64: &EmlMasterNode<Complex64> = unsafe {
                        &*(&self.root as *const EmlMasterNode<T> as *const EmlMasterNode<Complex64>)
                    };
                    let prediction = root_c64.eval(&vars, &params, &mut idx, temp);

                    // Safe access to y_target as Complex64
                    let y_c64: Complex64 = unsafe { std::mem::transmute_copy(&y_target) };
                    let loss = prediction
                        .sub(GenericVar::constant(y_c64))
                        .mul(prediction.sub(GenericVar::constant(y_c64)));
                    let adjoints = ad_gradient_c64(loss);
                    for (i, p) in params.iter().enumerate() {
                        total_grad[i] += adjoints[p.index].re
                            + if weights[i].abs() < 1.0 {
                                -self.entropy_weight * weights[i].signum()
                            } else {
                                0.0
                            };
                    }
                    ad_end();
                }
            }
            for i in 0..weights.len() {
                let g = total_grad[i] / dataset.len() as f64;
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                let m_hat = m[i] / (1.0 - 0.9_f64.powi(epoch as i32));
                let v_hat = v[i] / (1.0 - 0.999_f64.powi(epoch as i32));
                weights[i] -= current_lr * m_hat / (v_hat.sqrt() + 1e-8);
            }
            if epoch % 50 == 0 {
                let current_mse = self.calculate_mse(dataset);
                if current_mse < best_mse * 0.99 {
                    best_mse = current_mse;
                    plateau_count = 0;
                } else {
                    plateau_count += 1;
                    if plateau_count >= 3 {
                        current_lr *= 0.5;
                        plateau_count = 0;
                    }
                }
            }
        }
        let mut idx = 0;
        self.root.update_weights(&weights, &mut idx);
    }

    fn predict_inner(&self, x_vals: &[T]) -> T {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            ad_begin_f64();
            let vars: Vec<GenericVar<f64>> = x_vals
                .iter()
                .map(|v| GenericVar::<f64>::new(v.to_f64()))
                .collect();
            let mut weights = Vec::new();
            self.root.collect_weights(&mut weights);
            let params: Vec<GenericVar<f64>> =
                weights.iter().map(|&w| GenericVar::<f64>::new(w)).collect();
            let mut idx = 0;
            let root_f64: &EmlMasterNode<f64> =
                unsafe { &*(&self.root as *const EmlMasterNode<T> as *const EmlMasterNode<f64>) };
            let res = root_f64.eval(&vars, &params, &mut idx, 0.01).value;
            ad_end();
            T::from_f64(res)
        } else {
            ad_begin_c64();
            let vars: Vec<GenericVar<Complex64>> = x_vals
                .iter()
                .map(|v| {
                    let c: Complex64 = unsafe {
                        *(&v as *const &T as *const *const Complex64 as *const Complex64)
                    };
                    GenericVar::<Complex64>::new(c)
                })
                .collect();
            let mut weights = Vec::new();
            self.root.collect_weights(&mut weights);
            let params: Vec<GenericVar<Complex64>> = weights
                .iter()
                .map(|&w| GenericVar::<Complex64>::new(Complex64::new(w, 0.0)))
                .collect();
            let mut idx = 0;
            let root_c64: &EmlMasterNode<Complex64> = unsafe {
                &*(&self.root as *const EmlMasterNode<T> as *const EmlMasterNode<Complex64>)
            };
            let res = root_c64.eval(&vars, &params, &mut idx, 0.01).value;
            ad_end();
            unsafe { *(&res as *const Complex64 as *const T) }
        }
    }
}

impl EmlRegressor<f64> {
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.predict_inner(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eml_recovery_exp_f64() {
        let mut dataset = Vec::new();
        for i in 1..6 {
            let x = i as f64 * 0.5;
            dataset.push((vec![x], x.exp()));
        }
        let mut regressor = EmlRegressor::<f64>::new(0, 1);
        regressor.train(&dataset, 500);
        regressor.root.snap();
        for (x, y) in &dataset {
            assert!((regressor.predict(x) - *y).abs() < 1e-4);
        }
    }
}
