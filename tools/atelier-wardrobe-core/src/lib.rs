//! Exact small-instance oracle for user-declared wardrobe coverage tasks.
//!
//! The optimizer does not infer what a person should wear. Callers provide
//! explicit needs, compatible garment/repair options, user costs in one declared currency and a policy.
//! The result is a Pareto set; no universal style or sustainability score is
//! computed. This crate does not verify garment condition, fit or safety.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

pub const MAX_EXACT_OPTIONAL_CHOICES: usize = 18;
pub const MAX_EXACT_FRONTIER: usize = 10_000;
pub const MAX_NEEDS: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WardrobeNeed {
    pub id: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ChoiceOrigin {
    Owned,
    Repair,
    Alteration,
    Borrowed,
    Rented,
    SecondHand,
    Purchase,
    MadeToOrder,
}

impl ChoiceOrigin {
    fn adds_to_wardrobe(self) -> bool {
        matches!(
            self,
            Self::Borrowed | Self::Rented | Self::SecondHand | Self::Purchase | Self::MadeToOrder
        )
    }

    fn newly_manufactured(self) -> bool {
        matches!(self, Self::MadeToOrder)
    }
}

/// A garment or action that can cover one or more explicitly declared needs.
/// `must_keep` preserves a user-protected item in every plan, even when it is
/// not currently usable. Only `usable` options contribute need coverage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WardrobeChoice {
    pub id: String,
    pub covers_need_ids: BTreeSet<String>,
    pub origin: ChoiceOrigin,
    pub cost_minor_units: u64,
    pub currency_code: String,
    pub effort_minutes: u64,
    pub usable: bool,
    pub must_keep: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WardrobePolicy {
    /// False forbids any new-to-wardrobe acquisition, including borrowing,
    /// renting and second-hand acquisition. Owned items, repairs and
    /// alterations remain eligible.
    pub allow_additions: bool,
    /// Separately controls made-to-order production. A retail purchase is an
    /// acquisition, but is not automatically counted as newly manufactured
    /// because its production timing is not established by this model.
    pub allow_made_to_order: bool,
    /// ISO 4217-style uppercase currency code for every choice cost.
    /// The caller is responsible for confirming the code is currently valid.
    pub currency_code: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanMetrics {
    pub added_items: u64,
    pub newly_manufactured_items: u64,
    pub cost_minor_units: u64,
    pub currency_code: String,
    pub effort_minutes: u64,
    pub selected_choices: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WardrobePlan {
    pub selected_choice_ids: Vec<String>,
    pub covered_need_ids: Vec<String>,
    pub metrics: PlanMetrics,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactSearchResult {
    pub pareto_plans: Vec<WardrobePlan>,
    pub examined_subsets: u64,
    pub feasible_subsets: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WardrobeError {
    TooManyNeeds { actual: usize, maximum: usize },
    TooManyOptionalChoices { actual: usize, maximum: usize },
    EmptyNeedId,
    EmptyChoiceId,
    InvalidCurrencyCode(String),
    MixedCurrencyCodes { choice_id: String, expected: String, actual: String },
    DuplicateNeedId(String),
    DuplicateChoiceId(String),
    UnknownNeedReference { choice_id: String, need_id: String },
    ProtectedAdditionForbidden { choice_id: String },
    ProtectedProductionForbidden { choice_id: String },
    NoFeasiblePlan { uncovered_need_ids: Vec<String> },
    ArithmeticOverflow,
    FrontierLimitExceeded { maximum: usize },
}

/// Enumerate every subset of at most 18 optional choices, preserve every
/// `must_keep` choice, enforce user policy, and return the exact Pareto frontier
/// over acquisition count, made-to-order count, cost in the declared currency, effort and choice count.
/// Every metric is minimized. No weighted sum is imposed.
pub fn search_exact(
    needs: &[WardrobeNeed],
    choices: &[WardrobeChoice],
    policy: WardrobePolicy,
) -> Result<ExactSearchResult, WardrobeError> {
    if !is_currency_code_format(&policy.currency_code) {
        return Err(WardrobeError::InvalidCurrencyCode(policy.currency_code.clone()));
    }

    if needs.len() > MAX_NEEDS {
        return Err(WardrobeError::TooManyNeeds {
            actual: needs.len(),
            maximum: MAX_NEEDS,
        });
    }

    let mut need_bits = BTreeMap::new();
    for need in needs {
        if need.id.trim().is_empty() {
            return Err(WardrobeError::EmptyNeedId);
        }
        if need_bits.contains_key(&need.id) {
            return Err(WardrobeError::DuplicateNeedId(need.id.clone()));
        }
        let bit = need_bits.len();
        need_bits.insert(need.id.clone(), bit as u32);
    }

    let mut seen_choices = BTreeSet::new();
    let mut forced: Vec<&WardrobeChoice> = Vec::new();
    let mut optional: Vec<&WardrobeChoice> = Vec::new();

    for choice in choices {
        if choice.currency_code != policy.currency_code {
            return Err(WardrobeError::MixedCurrencyCodes {
                choice_id: choice.id.clone(),
                expected: policy.currency_code.clone(),
                actual: choice.currency_code.clone(),
            });
        }
        if choice.id.trim().is_empty() {
            return Err(WardrobeError::EmptyChoiceId);
        }
        if !seen_choices.insert(choice.id.as_str()) {
            return Err(WardrobeError::DuplicateChoiceId(choice.id.clone()));
        }
        for need_id in &choice.covers_need_ids {
            if !need_bits.contains_key(need_id) {
                return Err(WardrobeError::UnknownNeedReference {
                    choice_id: choice.id.clone(),
                    need_id: need_id.clone(),
                });
            }
        }

        if choice.must_keep {
            if !policy.allow_additions && choice.origin.adds_to_wardrobe() {
                return Err(WardrobeError::ProtectedAdditionForbidden {
                    choice_id: choice.id.clone(),
                });
            }
            if !policy.allow_made_to_order && choice.origin.newly_manufactured() {
                return Err(WardrobeError::ProtectedProductionForbidden {
                    choice_id: choice.id.clone(),
                });
            }
            forced.push(choice);
        } else if choice.usable
            && (policy.allow_additions || !choice.origin.adds_to_wardrobe())
            && (policy.allow_made_to_order || !choice.origin.newly_manufactured())
            && !choice.covers_need_ids.is_empty()
        {
            optional.push(choice);
        }
    }

    optional.sort_by(|left, right| left.id.cmp(&right.id));
    forced.sort_by(|left, right| left.id.cmp(&right.id));
    if optional.len() > MAX_EXACT_OPTIONAL_CHOICES {
        return Err(WardrobeError::TooManyOptionalChoices {
            actual: optional.len(),
            maximum: MAX_EXACT_OPTIONAL_CHOICES,
        });
    }

    let full_mask = if needs.len() == 64 {
        u64::MAX
    } else {
        (1_u64 << needs.len()) - 1
    };
    let forced_mask = coverage_mask(&forced, &need_bits);
    let forced_metrics = sum_metrics(&forced, &policy.currency_code)?;
    let subsets = 1_u64 << optional.len();
    let mut frontier: Vec<WardrobePlan> = Vec::new();
    let mut feasible_subsets = 0_u64;

    for subset in 0..subsets {
        let mut selected: Vec<&WardrobeChoice> = forced.clone();
        for (index, choice) in optional.iter().enumerate() {
            if (subset & (1_u64 << index)) != 0 {
                selected.push(choice);
            }
        }

        let mut covered = forced_mask;
        let mut metrics = forced_metrics.clone();
        for choice in selected.iter().skip(forced.len()) {
            covered |= choice_coverage_mask(choice, &need_bits);
            add_choice_metrics(&mut metrics, choice)?;
        }
        if covered != full_mask {
            continue;
        }
        feasible_subsets += 1;

        let mut ids: Vec<String> = selected.iter().map(|choice| choice.id.clone()).collect();
        ids.sort();
        let covered_need_ids: Vec<String> = needs
            .iter()
            .map(|need| need.id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let candidate = WardrobePlan {
            selected_choice_ids: ids,
            covered_need_ids,
            metrics,
        };

        if frontier.iter().any(|current| dominates(current, &candidate)) {
            continue;
        }
        frontier.retain(|current| !dominates(&candidate, current));
        frontier.push(candidate);
        if frontier.len() > MAX_EXACT_FRONTIER {
            return Err(WardrobeError::FrontierLimitExceeded {
                maximum: MAX_EXACT_FRONTIER,
            });
        }
    }

    if frontier.is_empty() {
        let coverable_mask = forced_mask | coverage_mask(&optional, &need_bits);
        let uncovered_need_ids = needs
            .iter()
            .enumerate()
            .filter(|(index, _)| coverable_mask & (1_u64 << *index) == 0)
            .map(|(_, need)| need.id.clone())
            .collect();
        return Err(WardrobeError::NoFeasiblePlan { uncovered_need_ids });
    }

    frontier.sort_by(|left, right| {
        left.metrics
            .newly_manufactured_items
            .cmp(&right.metrics.newly_manufactured_items)
            .then(left.metrics.added_items.cmp(&right.metrics.added_items))
            .then(left.metrics.cost_cents.cmp(&right.metrics.cost_cents))
            .then(left.metrics.effort_minutes.cmp(&right.metrics.effort_minutes))
            .then(left.metrics.selected_choices.cmp(&right.metrics.selected_choices))
            .then(left.selected_choice_ids.cmp(&right.selected_choice_ids))
    });

    Ok(ExactSearchResult {
        pareto_plans: frontier,
        examined_subsets: subsets,
        feasible_subsets,
    })
}

fn coverage_mask(
    choices: &[&WardrobeChoice],
    need_bits: &BTreeMap<String, u32>,
) -> u64 {
    choices.iter().fold(0_u64, |mask, choice| {
        if choice.usable {
            mask | choice_coverage_mask(choice, need_bits)
        } else {
            mask
        }
    })
}

fn choice_coverage_mask(
    choice: &WardrobeChoice,
    need_bits: &BTreeMap<String, u32>,
) -> u64 {
    if !choice.usable {
        return 0;
    }
    choice.covers_need_ids.iter().fold(0_u64, |mask, need_id| {
        mask | (1_u64 << need_bits[need_id])
    })
}

fn sum_metrics(choices: &[&WardrobeChoice], currency_code: &str) -> Result<PlanMetrics, WardrobeError> {
    let mut metrics = PlanMetrics {
        added_items: 0,
        newly_manufactured_items: 0,
        cost_minor_units: 0,
        currency_code: currency_code.to_owned(),
        effort_minutes: 0,
        selected_choices: 0,
    };
    for choice in choices {
        add_choice_metrics(&mut metrics, choice)?;
    }
    Ok(metrics)
}

fn add_choice_metrics(
    metrics: &mut PlanMetrics,
    choice: &WardrobeChoice,
) -> Result<(), WardrobeError> {
    metrics.added_items = metrics
        .added_items
        .checked_add(if choice.origin.adds_to_wardrobe() { 1 } else { 0 })
        .ok_or(WardrobeError::ArithmeticOverflow)?;
    metrics.newly_manufactured_items = metrics
        .newly_manufactured_items
        .checked_add(if choice.origin.newly_manufactured() { 1 } else { 0 })
        .ok_or(WardrobeError::ArithmeticOverflow)?;
    metrics.cost_minor_units = metrics
        .cost_minor_units
        .checked_add(choice.cost_cents)
        .ok_or(WardrobeError::ArithmeticOverflow)?;
    metrics.effort_minutes = metrics
        .effort_minutes
        .checked_add(choice.effort_minutes)
        .ok_or(WardrobeError::ArithmeticOverflow)?;
    metrics.selected_choices = metrics
        .selected_choices
        .checked_add(1)
        .ok_or(WardrobeError::ArithmeticOverflow)?;
    Ok(())
}

fn is_currency_code_format(value: &str) -> bool {
    value.len() == 3 && value.bytes().all(|byte| byte.is_ascii_uppercase())
}

fn dominates(left: &WardrobePlan, right: &WardrobePlan) -> bool {
    let a = &left.metrics;
    let b = &right.metrics;
    let no_worse = a.added_items <= b.added_items
        && a.newly_manufactured_items <= b.newly_manufactured_items
        && a.cost_minor_units <= b.cost_minor_units
        && a.effort_minutes <= b.effort_minutes
        && a.selected_choices <= b.selected_choices;
    let strictly_better = a.added_items < b.added_items
        || a.newly_manufactured_items < b.newly_manufactured_items
        || a.cost_minor_units < b.cost_minor_units
        || a.effort_minutes < b.effort_minutes
        || a.selected_choices < b.selected_choices;
    no_worse && strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;

    fn need(id: &str) -> WardrobeNeed {
        WardrobeNeed { id: id.to_owned() }
    }

    fn choice(
        id: &str,
        covers: &[&str],
        origin: ChoiceOrigin,
        cost_minor_units: u64,
        effort_minutes: u64,
    ) -> WardrobeChoice {
        WardrobeChoice {
            id: id.to_owned(),
            covers_need_ids: covers.iter().map(|value| (*value).to_owned()).collect(),
            origin,
            cost_minor_units,
            effort_minutes,
            usable: true,
            must_keep: false,
        }
    }

    #[test]
    fn existing_wardrobe_can_cover_needs_without_acquisition() {
        let needs = vec![need("work"), need("casual")];
        let choices = vec![
            choice("overshirt", &["work", "casual"], ChoiceOrigin::Owned, 0, 0),
            choice("new-jacket", &["work", "casual"], ChoiceOrigin::Purchase, 9000, 30),
        ];
        let result = search_exact(
            &needs,
            &choices,
            WardrobePolicy { allow_additions: false, allow_made_to_order: false, currency_code: "ZAR".to_owned() },
        )
        .unwrap();
        assert_eq!(result.pareto_plans.len(), 1);
        assert_eq!(result.pareto_plans[0].selected_choice_ids, vec!["overshirt"]);
        assert_eq!(result.pareto_plans[0].metrics.added_items, 0);
    }

    #[test]
    fn no_acquisition_policy_never_recommends_a_purchase() {
        let needs = vec![need("rain")];
        let choices = vec![choice("raincoat", &["rain"], ChoiceOrigin::Purchase, 6000, 15)];
        let result = search_exact(
            &needs,
            &choices,
            WardrobePolicy { allow_additions: false, allow_made_to_order: false, currency_code: "ZAR".to_owned() },
        );
        assert_eq!(
            result,
            Err(WardrobeError::NoFeasiblePlan {
                uncovered_need_ids: vec!["rain".to_owned()],
            })
        );
    }

    #[test]
    fn repair_and_purchase_alternatives_remain_on_the_pareto_frontier() {
        let needs = vec![need("weather-protection")];
        let choices = vec![
            choice("repair-existing-coat", &["weather-protection"], ChoiceOrigin::Repair, 2500, 90),
            choice("buy-new-coat", &["weather-protection"], ChoiceOrigin::Purchase, 8000, 20),
        ];
        let result = search_exact(
            &needs,
            &choices,
            WardrobePolicy { allow_additions: true, allow_made_to_order: true, currency_code: "ZAR".to_owned() },
        )
        .unwrap();
        assert_eq!(result.pareto_plans.len(), 2);
        assert!(result.pareto_plans.iter().any(|plan| plan.metrics.added_items == 0));
        assert!(result.pareto_plans.iter().any(|plan| plan.metrics.added_items == 1));
    }

    #[test]
    fn protected_unusable_garment_is_preserved_but_does_not_cover_need() {
        let needs = vec![need("formal")];
        let mut protected = choice("archive-jacket", &["formal"], ChoiceOrigin::Owned, 0, 0);
        protected.usable = false;
        protected.must_keep = true;
        let formal = choice("formal-option", &["formal"], ChoiceOrigin::Owned, 0, 0);
        let result = search_exact(
            &needs,
            &[protected, formal],
            WardrobePolicy { allow_additions: false, allow_made_to_order: false, currency_code: "ZAR".to_owned() },
        )
        .unwrap();
        assert_eq!(result.pareto_plans.len(), 1);
        assert!(result.pareto_plans[0].selected_choice_ids.contains(&"archive-jacket".to_owned()));
        assert!(result.pareto_plans[0].selected_choice_ids.contains(&"formal-option".to_owned()));
    }

    #[test]
    fn unknown_need_reference_is_rejected() {
        let result = search_exact(
            &[need("office")],
            &[choice("shirt", &["wedding"], ChoiceOrigin::Owned, 0, 0)],
            WardrobePolicy { allow_additions: false, allow_made_to_order: false, currency_code: "ZAR".to_owned() },
        );
        assert_eq!(
            result,
            Err(WardrobeError::UnknownNeedReference {
                choice_id: "shirt".to_owned(),
                need_id: "wedding".to_owned(),
            })
        );
    }

    #[test]
    fn exact_search_limit_fails_instead_of_returning_a_partial_answer() {
        let needs = vec![need("daily")];
        let choices: Vec<WardrobeChoice> = (0..=MAX_EXACT_OPTIONAL_CHOICES)
            .map(|index| {
                choice(
                    &format!("choice-{index:02}"),
                    &["daily"],
                    ChoiceOrigin::Owned,
                    index as u64,
                    index as u64,
                )
            })
            .collect();
        assert!(matches!(
            search_exact(
                &needs,
                &choices,
                WardrobePolicy { allow_additions: false, allow_made_to_order: false, currency_code: "ZAR".to_owned() }
            ),
            Err(WardrobeError::TooManyOptionalChoices { .. })
        ));
    }

    #[test]
    fn protected_new_acquisition_conflicts_with_no_additions_policy() {
        let needs = vec![need("daily")];
        let mut protected = choice("new-coat", &["daily"], ChoiceOrigin::Purchase, 5000, 10);
        protected.must_keep = true;
        assert_eq!(
            search_exact(
                &needs,
                &[protected],
                WardrobePolicy { allow_additions: false, allow_made_to_order: false, currency_code: "ZAR".to_owned() }
            ),
            Err(WardrobeError::ProtectedAdditionForbidden {
                choice_id: "new-coat".to_owned(),
            })
        );
    }

    #[test]
    fn made_to_order_can_be_blocked_without_blocking_other_additions() {
        let needs = vec![need("weather")];
        let choices = vec![
            choice("second-hand-coat", &["weather"], ChoiceOrigin::SecondHand, 1800, 30),
            choice("custom-coat", &["weather"], ChoiceOrigin::MadeToOrder, 7000, 60),
        ];
        let result = search_exact(
            &needs,
            &choices,
            WardrobePolicy { allow_additions: true, allow_made_to_order: false, currency_code: "ZAR".to_owned() },
        )
        .unwrap();
        assert_eq!(result.pareto_plans.len(), 1);
        assert_eq!(result.pareto_plans[0].selected_choice_ids, vec!["second-hand-coat"]);
        assert_eq!(result.pareto_plans[0].metrics.newly_manufactured_items, 0);
    }

    #[test]
    fn no_feasible_plan_reports_needs_uncovered_by_all_allowed_choices() {
        let needs = vec![need("office"), need("rain")];
        let choices = vec![choice("office-shirt", &["office"], ChoiceOrigin::Owned, 0, 0)];
        assert_eq!(
            search_exact(
                &needs,
                &choices,
                WardrobePolicy { allow_additions: false, allow_made_to_order: false, currency_code: "ZAR".to_owned() }
            ),
            Err(WardrobeError::NoFeasiblePlan {
                uncovered_need_ids: vec!["rain".to_owned()],
            })
        );
    }

    #[test]
    fn mixed_currency_choices_are_rejected() {
        let needs = vec![need("work")];
        let mut mixed = choice("shirt", &["work"], ChoiceOrigin::Owned, 2500, 10);
        mixed.currency_code = "USD".to_owned();
        assert!(matches!(
            search_exact(
                &needs,
                &[mixed],
                WardrobePolicy {
                    allow_additions: false,
                    allow_made_to_order: false,
                    currency_code: "ZAR".to_owned(),
                },
            ),
            Err(WardrobeError::MixedCurrencyCodes { .. })
        ));
    }

    #[test]
    fn malformed_currency_code_is_rejected() {
        let result = search_exact(
            &[need("work")],
            &[],
            WardrobePolicy {
                allow_additions: false,
                allow_made_to_order: false,
                currency_code: "zar".to_owned(),
            },
        );
        assert_eq!(result, Err(WardrobeError::InvalidCurrencyCode("zar".to_owned())));
    }

    #[test]
    fn empty_required_needs_still_preserve_protected_items() {
        let protected = WardrobeChoice {
            id: "archive-coat".to_owned(),
            covers_need_ids: BTreeSet::new(),
            origin: ChoiceOrigin::Owned,
            cost_minor_units: 0,
            currency_code: "ZAR".to_owned(),
            effort_minutes: 0,
            usable: false,
            must_keep: true,
        };
        let result = search_exact(
            &[],
            &[protected],
            WardrobePolicy { allow_additions: false, allow_made_to_order: false, currency_code: "ZAR".to_owned() },
        )
        .unwrap();
        assert_eq!(result.pareto_plans[0].selected_choice_ids, vec!["archive-coat"]);
    }
}
