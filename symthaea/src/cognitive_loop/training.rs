//! Async training infrastructure for background BPTT/SPSA training.
//!
//! The [`AsyncTrainerHandle`] runs a background thread that receives
//! [`TrainingSample`]s from the inference loop and periodically publishes
//! updated weights back, ensuring inference never blocks on training.

use std::sync::mpsc;
use ndarray::Array1;
use crate::dynamics::cfc::CfCNetwork;
use super::TrainingMethod;

/// A single training sample sent from the inference thread to the trainer.
pub(super) struct TrainingSample {
    pub input: Array1<f32>,
    pub target: Array1<f32>,
    pub dt: f32,
    pub learning_rate: f32,
    pub method: TrainingMethod,
    pub avg_loss: f32,
}

/// Handle held by `CognitiveLoopService` to communicate with the background
/// training thread.  Dropping it causes the background thread to exit.
///
/// The `Mutex<Receiver>` makes this struct `Sync` so that `CognitiveLoopService`
/// can implement `MetricsProvider: Send + Sync`.  In practice the mutex is
/// uncontended because `cycle()` is the only reader.
pub(super) struct AsyncTrainerHandle {
    pub sample_tx: mpsc::SyncSender<TrainingSample>,
    pub weights_rx: std::sync::Mutex<mpsc::Receiver<Vec<f32>>>,
    pub updates_applied: u64,
}

impl AsyncTrainerHandle {
    pub fn spawn(mut network: CfCNetwork) -> Self {
        let (sample_tx, sample_rx) = mpsc::sync_channel::<TrainingSample>(4);
        let (weights_tx, weights_rx) = mpsc::channel::<Vec<f32>>();

        std::thread::Builder::new()
            .name("symthaea-trainer".into())
            .spawn(move || {
                let mut steps_since_publish: u32 = 0;
                while let Ok(sample) = sample_rx.recv() {
                    let result = match sample.method {
                        TrainingMethod::Spsa => {
                            network.train_step_spsa(&sample.input, &sample.target, sample.dt, sample.learning_rate)
                        }
                        TrainingMethod::Bptt => {
                            network.train_step_bptt(&[sample.input], &[sample.target], &[sample.dt], sample.learning_rate)
                        }
                        TrainingMethod::BpttWithSpsaFallback => {
                            let bptt = network.train_step_bptt(
                                &[sample.input.clone()], &[sample.target.clone()],
                                &[sample.dt], sample.learning_rate,
                            );
                            match bptt {
                                Ok(loss) if loss.is_finite() && (sample.avg_loss <= 0.0 || loss < sample.avg_loss * 2.0) => Ok(loss),
                                _ => network.train_step_spsa(&sample.input, &sample.target, sample.dt, sample.learning_rate),
                            }
                        }
                    };
                    steps_since_publish += 1;
                    if steps_since_publish >= 4 && result.is_ok() {
                        let _ = weights_tx.send(network.get_weights());
                        steps_since_publish = 0;
                    }
                }
            })
            .expect("failed to spawn trainer thread");

        Self { sample_tx, weights_rx: std::sync::Mutex::new(weights_rx), updates_applied: 0 }
    }

    pub fn apply_latest_weights(&mut self, network: &mut CfCNetwork) -> bool {
        let mut latest: Option<Vec<f32>> = None;
        let rx = self.weights_rx.get_mut().expect("weights_rx mutex poisoned");
        while let Ok(w) = rx.try_recv() {
            latest = Some(w);
        }
        if let Some(w) = latest {
            network.set_weights(&w);
            self.updates_applied += 1;
            true
        } else {
            false
        }
    }

    pub fn send(&self, sample: TrainingSample) {
        let _ = self.sample_tx.try_send(sample);
    }
}
