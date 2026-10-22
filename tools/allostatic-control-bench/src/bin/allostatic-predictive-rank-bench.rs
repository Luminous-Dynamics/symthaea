//! Compare repeated point thresholds and fixed-horizon thresholds with predictive-rank martingales.
//!
//! Usage: allostatic-predictive-rank-bench <raw.csv> <summary.csv>
//! The score stream is synthetic. The predictive-rank monitor follows the
//! fixed-reference rank-martingale construction; exact guarantees remain
//! conditional on the algorithm's sampling/exchangeability assumptions.

use allostatic_control_bench::drift_monitor::calibrate_drift_threshold;
use allostatic_control_bench::rank_martingale::{
    RankMartingalePortfolio, ScoreToken,
};
use std::error::Error;
use std::fs;

const TARGET_FALSE_ALARM_RATE: f64 = 0.05;
const CALIBRATION_EPISODES: u64 = 159;
const EVALUATION_EPISODES: u64 = 300;
const HORIZON: usize = 256;
const ONSET_STEPS: [usize; 4] = [1, 17, 65, 193];
const SHIFT_MAGNITUDES: [f64; 3] = [0.04, 0.10, 0.20];
const SIGNAL_MODES: [&str; 4] = ["up_shift", "down_shift", "scale_shift", "sensor_blind"];

const RAW_HEADER: &str = "signal_mode,shift_magnitude,onset_step,episode_seed,pointwise_threshold,episode_max_threshold,pointwise_alarm_step,episode_max_alarm_step,prm_order_alarm_step,prm_dispersion_alarm_step,prm_portfolio_alarm_step,max_prm_portfolio_wealth";
const SUMMARY_HEADER: &str = "signal_mode,shift_magnitude,onset_step,episodes,pointwise_pre_shift_alarms,episode_max_pre_shift_alarms,prm_order_pre_shift_alarms,prm_dispersion_pre_shift_alarms,prm_portfolio_pre_shift_alarms,pointwise_post_shift_alarms,episode_max_post_shift_alarms,prm_order_post_shift_alarms,prm_dispersion_post_shift_alarms,prm_portfolio_post_shift_alarms,pointwise_no_post_shift_alarm,episode_max_no_post_shift_alarm,prm_order_no_post_shift_alarm,prm_dispersion_no_post_shift_alarm,prm_portfolio_no_post_shift_alarm,mean_pointwise_alarm_delay,mean_episode_max_alarm_delay,mean_prm_order_alarm_delay,mean_prm_dispersion_alarm_delay,mean_prm_portfolio_alarm_delay";

#[derive(Clone, Copy, Debug)]
struct EpisodeRecord {
    mode: usize,
    magnitude: f64,
    onset_step: usize,
    seed: u64,
    pointwise_threshold: f64,
    episode_max_threshold: f64,
    pointwise_alarm_step: usize,
    episode_max_alarm_step: usize,
    prm_order_alarm_step: usize,
    prm_dispersion_alarm_step: usize,
    prm_portfolio_alarm_step: usize,
    max_portfolio_wealth: f64,
}

impl EpisodeRecord {
    fn alarm_steps(self) -> [usize; 5] {
        [
            self.pointwise_alarm_step,
            self.episode_max_alarm_step,
            self.prm_order_alarm_step,
            self.prm_dispersion_alarm_step,
            self.prm_portfolio_alarm_step,
        ]
    }
}

fn unit_interval(seed: u64, step: usize, lane: u64) -> f64 {
    let mut value = seed
        .wrapping_add((step as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(lane);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value >> 11) as f64 / ((1_u64 << 53) as f64)
}

fn nominal_score(seed: u64, step: usize) -> f64 {
    0.4 + unit_interval(seed, step, 0x7261_6e6b_7363_6f72) * 0.2
}

fn score_token(seed: u64, step: usize, score: f64) -> Option<ScoreToken> {
    let tie_breaker = unit_interval(seed, step, 0x7469_655f_6272_6561);
    ScoreToken::new(score, tie_breaker)
}

fn shifted_score(base: f64, mode: usize, magnitude: f64, onset_step: usize, step: usize) -> f64 {
    if step + 1 < onset_step || mode == 3 {
        return base;
    }

    match mode {
        0 => base + magnitude,
        1 => base - magnitude,
        2 => 0.5 + (base - 0.5) * (1.0 + 4.0 * magnitude),
        _ => base,
    }
}

fn upper_quantile(values: &[f64], alpha: f64) -> Option<f64> {
    if values.is_empty()
        || !alpha.is_finite()
        || !(0.0..1.0).contains(&alpha)
        || values.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let rank = (((values.len() + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    if rank == 0 || rank > values.len() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered.get(rank - 1).copied()
}

fn calibrate_reference() -> Result<(Vec<ScoreToken>, f64, f64), Box<dyn Error>> {
    let mut reference = Vec::with_capacity(CALIBRATION_EPISODES as usize);
    let mut maxima = Vec::with_capacity(CALIBRATION_EPISODES as usize);

    for episode in 0..CALIBRATION_EPISODES {
        let seed = 100_000 + episode;
        let calibration_score = nominal_score(seed, 0);
        reference.push(score_token(seed, 0, calibration_score).ok_or("invalid reference token")?);

        let stream_seed = 400_000 + episode;
        let mut maximum: f64 = 0.0;
        for step in 0..HORIZON {
            maximum = maximum.max(nominal_score(stream_seed, step));
        }
        maxima.push(maximum);
    }

    let reference_scores: Vec<f64> = reference.iter().map(|token| token.score()).collect();
    let pointwise_threshold = upper_quantile(&reference_scores, TARGET_FALSE_ALARM_RATE)
        .ok_or("failed to calibrate pointwise threshold")?;
    let episode_max_threshold = calibrate_drift_threshold(&maxima, TARGET_FALSE_ALARM_RATE)
        .ok_or("failed to calibrate episode-maximum threshold")?
        .score_threshold();

    Ok((reference, pointwise_threshold, episode_max_threshold))
}

fn run_episode(
    mode: usize,
    magnitude: f64,
    onset_step: usize,
    episode: u64,
    reference: &[ScoreToken],
    pointwise_threshold: f64,
    episode_max_threshold: f64,
) -> Result<EpisodeRecord, Box<dyn Error>> {
    let seed = 200_000 + episode;
    let mut portfolio =
        RankMartingalePortfolio::new(reference, TARGET_FALSE_ALARM_RATE)
            .ok_or("failed to initialize predictive-rank portfolio")?;

    let mut pointwise_alarm_step = 0;
    let mut episode_max_alarm_step = 0;
    let mut prm_order_alarm_step = 0;
    let mut prm_dispersion_alarm_step = 0;
    let mut prm_portfolio_alarm_step = 0;
    let mut max_portfolio_wealth: f64 = 1.0;

    for step in 0..HORIZON {
        let base = nominal_score(seed, step);
        let score = shifted_score(base, mode, magnitude, onset_step, step);
        let token = score_token(seed, step, score).ok_or("invalid test token")?;
        let control_step = step + 1;

        if score > pointwise_threshold && pointwise_alarm_step == 0 {
            pointwise_alarm_step = control_step;
        }
        if score > episode_max_threshold && episode_max_alarm_step == 0 {
            episode_max_alarm_step = control_step;
        }

        let observation = portfolio.observe(token).ok_or("invalid rank-martingale update")?;
        if observation.order().alarm() && prm_order_alarm_step == 0 {
            prm_order_alarm_step = control_step;
        }
        if observation.dispersion().alarm() && prm_dispersion_alarm_step == 0 {
            prm_dispersion_alarm_step = control_step;
        }
        if observation.portfolio_alarm() && prm_portfolio_alarm_step == 0 {
            prm_portfolio_alarm_step = control_step;
        }
        max_portfolio_wealth = max_portfolio_wealth.max(observation.portfolio_wealth());
    }

    Ok(EpisodeRecord {
        mode,
        magnitude,
        onset_step,
        seed,
        pointwise_threshold,
        episode_max_threshold,
        pointwise_alarm_step,
        episode_max_alarm_step,
        prm_order_alarm_step,
        prm_dispersion_alarm_step,
        prm_portfolio_alarm_step,
        max_portfolio_wealth,
    })
}

fn write_raw(path: &str, records: &[EpisodeRecord]) -> Result<(), Box<dyn Error>> {
    let mut output = String::with_capacity(records.len() * 140);
    output.push_str(RAW_HEADER);
    output.push('\n');
    for record in records {
        output.push_str(&format!(
            "{},{:.2},{},{},{:.9},{:.9},{},{},{},{},{},{:.12e}\n",
            SIGNAL_MODES[record.mode],
            record.magnitude,
            record.onset_step,
            record.seed,
            record.pointwise_threshold,
            record.episode_max_threshold,
            record.pointwise_alarm_step,
            record.episode_max_alarm_step,
            record.prm_order_alarm_step,
            record.prm_dispersion_alarm_step,
            record.prm_portfolio_alarm_step,
            record.max_portfolio_wealth,
        ));
    }
    fs::write(path, output)?;
    Ok(())
}

fn mean_delay(records: &[EpisodeRecord], detector: usize) -> String {
    let mut total_delay = 0_usize;
    let mut count = 0_usize;
    for record in records {
        let alarm = record.alarm_steps()[detector];
        if alarm >= record.onset_step && alarm > 0 {
            total_delay += alarm - record.onset_step;
            count += 1;
        }
    }
    if count == 0 {
        "NA".to_owned()
    } else {
        format!("{:.6}", total_delay as f64 / count as f64)
    }
}

fn write_summary(path: &str, records: &[EpisodeRecord]) -> Result<(), Box<dyn Error>> {
    let mut output = String::new();
    output.push_str(SUMMARY_HEADER);
    output.push('\n');

    for mode in 0..SIGNAL_MODES.len() {
        for magnitude in SHIFT_MAGNITUDES {
            for onset_step in ONSET_STEPS {
                let cell: Vec<EpisodeRecord> = records
                    .iter()
                    .copied()
                    .filter(|record| {
                        record.mode == mode
                            && (record.magnitude - magnitude).abs() < f64::EPSILON
                            && record.onset_step == onset_step
                    })
                    .collect();
                let pre = |detector: usize| {
                    cell.iter()
                        .filter(|record| {
                            let alarm = record.alarm_steps()[detector];
                            alarm > 0 && alarm < onset_step
                        })
                        .count()
                };
                let post = |detector: usize| {
                    cell.iter()
                        .filter(|record| record.alarm_steps()[detector] >= onset_step)
                        .count()
                };

                let post_counts: Vec<usize> = (0..5).map(post).collect();
                let no_post: Vec<usize> = post_counts
                    .iter()
                    .map(|count| EVALUATION_EPISODES as usize - count)
                    .collect();

                output.push_str(&format!(
                    "{},{:.2},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    SIGNAL_MODES[mode],
                    magnitude,
                    onset_step,
                    cell.len(),
                    pre(0),
                    pre(1),
                    pre(2),
                    pre(3),
                    pre(4),
                    post_counts[0],
                    post_counts[1],
                    post_counts[2],
                    post_counts[3],
                    post_counts[4],
                    no_post[0],
                    no_post[1],
                    no_post[2],
                    no_post[3],
                    no_post[4],
                    mean_delay(&cell, 0),
                    mean_delay(&cell, 1),
                    mean_delay(&cell, 2),
                    mean_delay(&cell, 3),
                    mean_delay(&cell, 4),
                ));
            }
        }
    }

    fs::write(path, output)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: allostatic-predictive-rank-bench <raw.csv> <summary.csv>");
        std::process::exit(2);
    }

    let (reference, pointwise_threshold, episode_max_threshold) = calibrate_reference()?;
    let mut records = Vec::with_capacity(
        SIGNAL_MODES.len()
            * SHIFT_MAGNITUDES.len()
            * ONSET_STEPS.len()
            * EVALUATION_EPISODES as usize,
    );

    for mode in 0..SIGNAL_MODES.len() {
        for magnitude in SHIFT_MAGNITUDES {
            for onset_step in ONSET_STEPS {
                for episode in 0..EVALUATION_EPISODES {
                    records.push(run_episode(
                        mode,
                        magnitude,
                        onset_step,
                        episode,
                        &reference,
                        pointwise_threshold,
                        episode_max_threshold,
                    )?);
                }
            }
        }
    }

    write_raw(&args[1], &records)?;
    write_summary(&args[2], &records)?;
    println!("Predictive-rank raw results: {}", args[1]);
    println!("Predictive-rank summary: {}", args[2]);
    println!("Episodes: {}", records.len());
    Ok(())
}
