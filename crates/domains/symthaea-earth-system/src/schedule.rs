// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Event-aligned integration schedules for piecewise forcing protocols.
//!
//! A fixed RK4 step that straddles a pulse edge samples two different regimes
//! inside one polynomial step. These schedules split intervals at every known
//! protocol breakpoint while retaining the caller's nominal maximum step.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};

/// Hard bound preventing accidental allocation of an unreasonably large grid.
pub const MAX_SCHEDULE_INTERVALS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrationInterval {
    pub start: f64,
    pub end: f64,
}

impl IntegrationInterval {
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Build intervals over `[0, duration]` with no interval longer than
/// `nominal_step`, splitting exactly at finite events inside the domain.
pub fn event_aligned_intervals(
    duration: f64,
    nominal_step: f64,
    events: &[f64],
) -> Result<Vec<IntegrationInterval>, ModelError> {
    require_non_negative("duration", duration)?;
    require_positive("nominal_step", nominal_step)?;
    for &event in events {
        require_finite("event_time", event)?;
        if event < 0.0 {
            return Err(ModelError::OutOfRange {
                parameter: "event_time",
                value: event,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
    }
    if events.len() > MAX_SCHEDULE_INTERVALS {
        return Err(ModelError::ScheduleTooLarge {
            requested: events.len(),
            maximum: MAX_SCHEDULE_INTERVALS,
        });
    }
    if duration == 0.0 {
        return Ok(Vec::new());
    }

    let base_count = (duration / nominal_step).ceil();
    if !base_count.is_finite() || base_count > MAX_SCHEDULE_INTERVALS as f64 {
        return Err(ModelError::ScheduleTooLarge {
            requested: if base_count.is_finite() {
                base_count as usize
            } else {
                usize::MAX
            },
            maximum: MAX_SCHEDULE_INTERVALS,
        });
    }

    let requested =
        (base_count as usize)
            .checked_add(events.len())
            .ok_or(ModelError::ScheduleTooLarge {
                requested: usize::MAX,
                maximum: MAX_SCHEDULE_INTERVALS,
            })?;
    if requested > MAX_SCHEDULE_INTERVALS {
        return Err(ModelError::ScheduleTooLarge {
            requested,
            maximum: MAX_SCHEDULE_INTERVALS,
        });
    }
    let mut boundaries = Vec::with_capacity(requested + 1);
    boundaries.push(0.0);
    let mut index = 1usize;
    loop {
        let time = index as f64 * nominal_step;
        if time >= duration {
            break;
        }
        boundaries.push(time);
        index += 1;
    }
    for &event in events {
        if event > 0.0 && event < duration {
            boundaries.push(event);
        }
    }
    boundaries.push(duration);
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup_by(|left, right| {
        let scale = left.abs().max(right.abs()).max(1.0);
        (*left - *right).abs() <= 16.0 * f64::EPSILON * scale
    });

    if boundaries.len().saturating_sub(1) > MAX_SCHEDULE_INTERVALS {
        return Err(ModelError::ScheduleTooLarge {
            requested: boundaries.len() - 1,
            maximum: MAX_SCHEDULE_INTERVALS,
        });
    }

    Ok(boundaries
        .windows(2)
        .map(|pair| IntegrationInterval {
            start: pair[0],
            end: pair[1],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_splits_at_off_grid_events() {
        let intervals = event_aligned_intervals(10.0, 4.0, &[2.5, 7.25]).unwrap();
        let endpoints: Vec<_> = intervals.iter().map(|step| step.end).collect();
        assert_eq!(endpoints, vec![2.5, 4.0, 7.25, 8.0, 10.0]);
        assert!(intervals.iter().all(|step| step.duration() <= 4.0));
    }

    #[test]
    fn zero_duration_has_no_intervals() {
        assert!(
            event_aligned_intervals(0.0, 1.0, &[0.5])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn excessive_event_count_fails_before_allocation() {
        let events = vec![0.5; MAX_SCHEDULE_INTERVALS + 1];
        assert!(matches!(
            event_aligned_intervals(1.0, 1.0, &events),
            Err(ModelError::ScheduleTooLarge { .. })
        ));
    }
}
