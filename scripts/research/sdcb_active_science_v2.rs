//! Active Scientific Intelligence v2: bounded linear-system reference.
 // Build: rustc --edition=2024 --test scripts/research/sdcb_active_science_v2.rs -o /tmp/sdcb-active-science-v2
 // Run:   /tmp/sdcb-active-science-v2
 //!
 //! Evaluator/reference primitive, not a production research agent. Hidden truth
 //! and the noise seed are separate from the candidate-visible episode.

const EPS: f64 = 1.0e-10;
const MIN_STATES: usize = 2;
const MAX_STATES: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActiveScienceError {
    InvalidEpisode, InvalidHypothesis, InvalidProbe, DuplicateIdentifier,
    UnknownProbe, UnknownTrueHypothesis, BudgetExceeded, InvalidReceipt,
    NumericalFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix { pub rows: usize, pub cols: usize, pub data: Vec<Vec<f64>> }

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self, ActiveScienceError> {
        if data.is_empty() || data[0].is_empty() { return Err(ActiveScienceError::InvalidHypothesis); }
        let cols = data[0].len();
        if data.iter().any(|r| r.len() != cols || r.iter().any(|x| !x.is_finite())) {
            return Err(ActiveScienceError::InvalidHypothesis);
        }
        Ok(Self { rows: data.len(), cols, data })
    }
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![vec![0.0; cols]; rows] }
    }
    fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>, ActiveScienceError> {
        if self.cols != x.len() || x.iter().any(|v| !v.is_finite()) {
            return Err(ActiveScienceError::InvalidHypothesis);
        }
        Ok(self.data.iter().map(|r| r.iter().zip(x).map(|(a,b)| a*b).sum()).collect())
    }
    fn mul(&self, rhs: &Matrix) -> Result<Matrix, ActiveScienceError> {
        if self.cols != rhs.rows { return Err(ActiveScienceError::InvalidHypothesis); }
        let mut out = Matrix::zeros(self.rows, rhs.cols);
        for i in 0..self.rows { for k in 0..self.cols { for j in 0..rhs.cols {
            out.data[i][j] += self.data[i][k] * rhs.data[k][j];
        }}}
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearHypothesis {
    pub hypothesis_id: String,
    /// x[t+1] = A x[t] + B u[t]
    pub a: Matrix,
    /// Single-input control vector.
    pub b: Vec<f64>,
    /// Rows are candidate sensors: y[t] = C x[t] + bounded noise.
    pub c: Matrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceVector {
    pub query_units: u32,
    pub experiment_units: u32,
    pub measurement_units: u32,
}
impl ResourceVector {
    fn fits(self, b: Self) -> bool {
        self.query_units <= b.query_units && self.experiment_units <= b.experiment_units
            && self.measurement_units <= b.measurement_units
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    pub probe_id: String,
    pub sensor_index: usize,
    /// One bounded scalar input per discrete transition.
    pub input_sequence: Vec<f64>,
    pub cost: ResourceVector,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEpisode {
    pub schema_version: String,
    pub episode_id: String,
    pub hypotheses: Vec<LinearHypothesis>,
    pub initial_state: Vec<f64>,
    pub passive_input_sequence: Vec<f64>,
    pub sensor_noise_bound: f64,
    pub maximum_absolute_input: f64,
    pub permitted_probes: Vec<Probe>,
    pub remaining_budget: ResourceVector,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatorSecret {
    /// Must never be serialized into CandidateEpisode.
    pub true_hypothesis_id: String,
    pub noise_seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationReceipt {
    pub episode_id: String,
    pub probe_id: String,
    pub sensor_index: usize,
    pub observed_trace: Vec<f64>,
    pub declared_noise_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeChoice {
    pub probe_id: String,
    pub minimum_pairwise_margin: f64,
    pub cost: ResourceVector,
}

fn validate_hypothesis(h: &LinearHypothesis, n: usize) -> Result<(), ActiveScienceError> {
    if h.hypothesis_id.trim().is_empty() || h.a.rows != n || h.a.cols != n
        || h.b.len() != n || h.c.cols != n || h.c.rows == 0
        || h.b.iter().any(|x| !x.is_finite()) {
        return Err(ActiveScienceError::InvalidHypothesis);
    }
    Ok(())
}

pub fn validate_episode(e: &CandidateEpisode) -> Result<(), ActiveScienceError> {
    let n = e.initial_state.len();
    if e.schema_version != "sdcb-active-science-v2" || e.episode_id.trim().is_empty()
        || !(MIN_STATES..=MAX_STATES).contains(&n)
        || e.initial_state.iter().any(|x| !x.is_finite())
        || e.passive_input_sequence.is_empty()
        || !e.sensor_noise_bound.is_finite() || e.sensor_noise_bound < 0.0
        || !e.maximum_absolute_input.is_finite() || e.maximum_absolute_input <= 0.0
        || e.passive_input_sequence.iter().any(|u| !u.is_finite() || u.abs() > e.maximum_absolute_input)
        || e.hypotheses.len() < 2 || e.permitted_probes.is_empty() {
        return Err(ActiveScienceError::InvalidEpisode);
    }
    let sensors = e.hypotheses[0].c.rows;
    for (i,h) in e.hypotheses.iter().enumerate() {
        validate_hypothesis(h, n)?;
        if h.c.rows != sensors { return Err(ActiveScienceError::InvalidHypothesis); }
        if e.hypotheses[..i].iter().any(|x| x.hypothesis_id == h.hypothesis_id) {
            return Err(ActiveScienceError::DuplicateIdentifier);
        }
    }
    for (i,p) in e.permitted_probes.iter().enumerate() {
        if p.probe_id.trim().is_empty() || p.sensor_index >= sensors || p.input_sequence.is_empty()
            || p.input_sequence.iter().any(|u| !u.is_finite() || u.abs() > e.maximum_absolute_input) {
            return Err(ActiveScienceError::InvalidProbe);
        }
        if e.permitted_probes[..i].iter().any(|x| x.probe_id == p.probe_id) {
            return Err(ActiveScienceError::DuplicateIdentifier);
        }
    }
    Ok(())
}

fn find_hypothesis<'a>(e: &'a CandidateEpisode, id: &str) -> Option<&'a LinearHypothesis> {
    e.hypotheses.iter().find(|h| h.hypothesis_id == id)
}
fn find_probe<'a>(e: &'a CandidateEpisode, id: &str) -> Option<&'a Probe> {
    e.permitted_probes.iter().find(|p| p.probe_id == id)
}

/// Trace contains y[0], then one observation after each input.
pub fn predict_trace(h: &LinearHypothesis, x0: &[f64], sensor: usize, inputs: &[f64])
    -> Result<Vec<f64>, ActiveScienceError> {
    let n = x0.len();
    validate_hypothesis(h, n)?;
    if sensor >= h.c.rows || inputs.is_empty() || inputs.iter().any(|u| !u.is_finite()) {
        return Err(ActiveScienceError::InvalidProbe);
    }
    let c = &h.c.data[sensor];
    let mut x = x0.to_vec();
    let mut trace = vec![c.iter().zip(&x).map(|(a,b)| a*b).sum()];
    for u in inputs {
        let mut next = h.a.mul_vec(&x)?;
        for (v,b) in next.iter_mut().zip(&h.b) { *v += b*u; }
        x = next;
        trace.push(c.iter().zip(&x).map(|(a,b)| a*b).sum());
    }
    if trace.iter().any(|v| !v.is_finite()) { return Err(ActiveScienceError::NumericalFailure); }
    Ok(trace)
}

/// Positive margin means every hypothesis pair has at least one time point
/// with disjoint bounded-noise intervals. This is not a general identifiability theorem.
pub fn minimum_pairwise_margin(e: &CandidateEpisode, p: &Probe) -> Result<f64, ActiveScienceError> {
    validate_episode(e)?;
    if find_probe(e, &p.probe_id) != Some(p) { return Err(ActiveScienceError::UnknownProbe); }
    let mut traces = Vec::new();
    for h in &e.hypotheses { traces.push(predict_trace(h, &e.initial_state, p.sensor_index, &p.input_sequence)?); }
    let mut minimum = f64::INFINITY;
    for i in 0..traces.len() { for j in (i+1)..traces.len() {
        if traces[i].len() != traces[j].len() { return Err(ActiveScienceError::InvalidHypothesis); }
        let separation = traces[i].iter().zip(&traces[j]).map(|(a,b)| (a-b).abs()).fold(0.0, f64::max);
        minimum = minimum.min(separation - 2.0*e.sensor_noise_bound);
    }}
    if !minimum.is_finite() { return Err(ActiveScienceError::NumericalFailure); }
    Ok(minimum)
}

/// Exact one-probe maximin choice over finite permitted offers. Abstains if no
/// affordable offer separates every candidate pair beyond the noise bound.
pub fn choose_discriminating_probe(e: &CandidateEpisode) -> Result<Option<ProbeChoice>, ActiveScienceError> {
    validate_episode(e)?;
    let mut best: Option<ProbeChoice> = None;
    for p in &e.permitted_probes {
        if !p.cost.fits(e.remaining_budget) { continue; }
        let margin = minimum_pairwise_margin(e,p)?;
        if margin <= EPS { continue; }
        let candidate = ProbeChoice { probe_id:p.probe_id.clone(), minimum_pairwise_margin:margin, cost:p.cost };
        let replace = match &best {
            None => true,
            Some(old) if margin > old.minimum_pairwise_margin + EPS => true,
            Some(old) if (margin-old.minimum_pairwise_margin).abs() <= EPS => candidate.probe_id < old.probe_id,
            _ => false,
        };
        if replace { best=Some(candidate); }
    }
    Ok(best)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 { state: u64 }
impl SplitMix64 {
    fn new(seed:u64)->Self { Self{state:seed} }
    fn next_u64(&mut self)->u64 {
        self.state=self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z=self.state;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        z^(z>>31)
    }
    fn symmetric_unit(&mut self)->f64 {
        let m=self.next_u64()>>11;
        (m as f64 / ((1u64<<53) as f64))*2.0-1.0
    }
}

/// Evaluator-only reference run. Production must authenticate and persist receipts.
pub fn execute_reference_probe(e:&CandidateEpisode, secret:&EvaluatorSecret, probe_id:&str)
    -> Result<ObservationReceipt,ActiveScienceError> {
    validate_episode(e)?;
    let p=find_probe(e,probe_id).ok_or(ActiveScienceError::UnknownProbe)?;
    if !p.cost.fits(e.remaining_budget) { return Err(ActiveScienceError::BudgetExceeded); }
    let h=find_hypothesis(e,&secret.true_hypothesis_id).ok_or(ActiveScienceError::UnknownTrueHypothesis)?;
    let predicted=predict_trace(h,&e.initial_state,p.sensor_index,&p.input_sequence)?;
    let mut rng=SplitMix64::new(secret.noise_seed);
    let observed_trace=predicted.into_iter().map(|v|v+rng.symmetric_unit()*e.sensor_noise_bound).collect();
    Ok(ObservationReceipt { episode_id:e.episode_id.clone(), probe_id:probe_id.into(),
        sensor_index:p.sensor_index, observed_trace, declared_noise_bound:e.sensor_noise_bound })
}

/// Set of candidate models compatible with the receipt and declared bounded noise.
/// This is not a proof that the true mechanism was present in the candidate set.
pub fn consistent_hypotheses(e:&CandidateEpisode,p:&Probe,r:&ObservationReceipt)
    -> Result<Vec<String>,ActiveScienceError> {
    validate_episode(e)?;
    if find_probe(e,&p.probe_id)!=Some(p) || r.episode_id!=e.episode_id || r.probe_id!=p.probe_id
        || r.sensor_index!=p.sensor_index || r.observed_trace.len()!=p.input_sequence.len()+1
        || !r.declared_noise_bound.is_finite()
        || (r.declared_noise_bound-e.sensor_noise_bound).abs()>EPS
        || r.observed_trace.iter().any(|v|!v.is_finite()) {
        return Err(ActiveScienceError::InvalidReceipt);
    }
    let mut out=Vec::new();
    for h in &e.hypotheses {
        let predicted=predict_trace(h,&e.initial_state,p.sensor_index,&p.input_sequence)?;
        if predicted.iter().zip(&r.observed_trace).all(|(a,b)|(a-b).abs()<=e.sensor_noise_bound+EPS) {
            out.push(h.hypothesis_id.clone());
        }
    }
    Ok(out)
}

pub fn matrix_rank(m:&Matrix)->Result<usize,ActiveScienceError> {
    if m.rows==0 || m.cols==0 || m.data.len()!=m.rows
        || m.data.iter().any(|r|r.len()!=m.cols||r.iter().any(|v|!v.is_finite())) {
        return Err(ActiveScienceError::InvalidHypothesis);
    }
    let mut a=m.data.clone();
    let mut pivot_row=0;
    for col in 0..m.cols {
        if pivot_row>=m.rows { break; }
        let pivot=(pivot_row..m.rows).max_by(|i,j|a[*i][col].abs().total_cmp(&a[*j][col].abs()));
        let Some(pivot)=pivot else { break; };
        if a[pivot][col].abs()<=EPS { continue; }
        a.swap(pivot_row,pivot);
        let d=a[pivot_row][col];
        for j in col..m.cols { a[pivot_row][j]/=d; }
        for i in 0..m.rows { if i!=pivot_row {
            let factor=a[i][col];
            for j in col..m.cols { a[i][j]-=factor*a[pivot_row][j]; }
        }}
        pivot_row+=1;
    }
    Ok(pivot_row)
}

/// Observability rank of (A,C_sensor): [C; CA; ...; CA^(n-1)].
pub fn observability_rank(h:&LinearHypothesis,sensor:usize)->Result<usize,ActiveScienceError> {
    let n=h.a.rows;
    validate_hypothesis(h,n)?;
    if sensor>=h.c.rows { return Err(ActiveScienceError::InvalidProbe); }
    let mut power=Matrix::zeros(n,n);
    for i in 0..n { power.data[i][i]=1.0; }
    let mut rows=Vec::with_capacity(n);
    for _ in 0..n {
        rows.push(h.c.mul(&power)?.data[sensor].clone());
        power=h.a.mul(&power)?;
    }
    matrix_rank(&Matrix::new(rows)?)
}

/// Controllability rank of (A,B): [B, AB, ... A^(n-1)B].
pub fn controllability_rank(h:&LinearHypothesis)->Result<usize,ActiveScienceError> {
    let n=h.a.rows;
    validate_hypothesis(h,n)?;
    let mut power=Matrix::zeros(n,n);
    for i in 0..n { power.data[i][i]=1.0; }
    let mut columns=vec![Vec::with_capacity(n);n];
    for _ in 0..n {
        let col=power.mul_vec(&h.b)?;
        for (i,v) in col.iter().enumerate() { columns[i].push(*v); }
        power=h.a.mul(&power)?;
    }
    matrix_rank(&Matrix::new(columns)?)
}

/// Frozen pair: passive zero-input outputs match; an impulse separates mechanisms.
pub fn observational_equivalence_fixture()->CandidateEpisode {
    let a=Matrix::new(vec![vec![0.0,1.0],vec![0.0,0.0]]).unwrap();
    let c=Matrix::new(vec![vec![1.0,0.0],vec![0.0,1.0]]).unwrap();
    let h1=LinearHypothesis{hypothesis_id:"mechanism-basis-x".into(),a:a.clone(),b:vec![1.0,0.0],c:c.clone()};
    let h2=LinearHypothesis{hypothesis_id:"mechanism-basis-y".into(),a,b:vec![0.0,1.0],c};
    CandidateEpisode {
        schema_version:"sdcb-active-science-v2".into(), episode_id:"observational-equivalence-2state-v1".into(),
        hypotheses:vec![h1,h2],initial_state:vec![0.0,0.0],passive_input_sequence:vec![0.0,0.0],
        sensor_noise_bound:0.05,maximum_absolute_input:1.0,
        permitted_probes:vec![
            Probe{probe_id:"no-op-sensor-x".into(),sensor_index:0,input_sequence:vec![0.0,0.0],
                cost:ResourceVector{query_units:1,experiment_units:0,measurement_units:1}},
            Probe{probe_id:"impulse-sensor-x".into(),sensor_index:0,input_sequence:vec![1.0,0.0],
                cost:ResourceVector{query_units:1,experiment_units:1,measurement_units:1}},
            Probe{probe_id:"impulse-sensor-y".into(),sensor_index:1,input_sequence:vec![1.0,0.0],
                cost:ResourceVector{query_units:1,experiment_units:1,measurement_units:1}},
        ],
        remaining_budget:ResourceVector{query_units:2,experiment_units:1,measurement_units:2},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passive_zero_input_cannot_identify_equivalent_mechanisms() {
        let e=observational_equivalence_fixture(); validate_episode(&e).unwrap();
        let a=predict_trace(&e.hypotheses[0],&e.initial_state,0,&e.passive_input_sequence).unwrap();
        let b=predict_trace(&e.hypotheses[1],&e.initial_state,0,&e.passive_input_sequence).unwrap();
        assert_eq!(a,b); assert_eq!(a,vec![0.0,0.0,0.0]);
    }
    #[test]
    fn intervention_breaks_passive_observational_equivalence() {
        let e=observational_equivalence_fixture(); let p=find_probe(&e,"impulse-sensor-x").unwrap();
        let a=predict_trace(&e.hypotheses[0],&e.initial_state,p.sensor_index,&p.input_sequence).unwrap();
        let b=predict_trace(&e.hypotheses[1],&e.initial_state,p.sensor_index,&p.input_sequence).unwrap();
        assert_eq!(a,vec![0.0,1.0,0.0]); assert_eq!(b,vec![0.0,0.0,1.0]);
    }
    #[test]
    fn maximin_selector_rejects_noop_and_selects_discriminating_probe() {
        let e=observational_equivalence_fixture(); let c=choose_discriminating_probe(&e).unwrap().unwrap();
        assert_eq!(c.probe_id,"impulse-sensor-x"); assert!((c.minimum_pairwise_margin-0.9).abs()<EPS);
    }
    #[test]
    fn no_experiment_budget_causes_abstention() {
        let mut e=observational_equivalence_fixture(); e.remaining_budget.experiment_units=0;
        assert_eq!(choose_discriminating_probe(&e).unwrap(),None);
    }
    #[test]
    fn reference_execution_keeps_truth_in_evaluator_secret() {
        let e=observational_equivalence_fixture();
        let s=EvaluatorSecret{true_hypothesis_id:"mechanism-basis-y".into(),noise_seed:42};
        let r=execute_reference_probe(&e,&s,"impulse-sensor-x").unwrap();
        let p=find_probe(&e,"impulse-sensor-x").unwrap();
        assert_eq!(consistent_hypotheses(&e,p,&r).unwrap(),vec!["mechanism-basis-y"]);
    }
    #[test]
    fn bounded_noise_receipt_never_excludes_the_true_hypothesis() {
        for seed in 0..128 {
            let e=observational_equivalence_fixture();
            let s=EvaluatorSecret{true_hypothesis_id:"mechanism-basis-x".into(),noise_seed:seed};
            let r=execute_reference_probe(&e,&s,"impulse-sensor-x").unwrap();
            let p=find_probe(&e,"impulse-sensor-x").unwrap();
            assert!(consistent_hypotheses(&e,p,&r).unwrap().contains(&s.true_hypothesis_id));
        }
    }
    #[test]
    fn unaffordable_probe_is_rejected_at_evaluator_boundary() {
        let mut e=observational_equivalence_fixture(); e.remaining_budget.experiment_units=0;
        let s=EvaluatorSecret{true_hypothesis_id:"mechanism-basis-x".into(),noise_seed:1};
        assert_eq!(execute_reference_probe(&e,&s,"impulse-sensor-x"),Err(ActiveScienceError::BudgetExceeded));
    }
    #[test]
    fn unknown_or_unpermitted_probe_is_rejected() {
        let e=observational_equivalence_fixture(); let s=EvaluatorSecret{true_hypothesis_id:"mechanism-basis-x".into(),noise_seed:1};
        assert_eq!(execute_reference_probe(&e,&s,"invented-probe"),Err(ActiveScienceError::UnknownProbe));
    }
    #[test]
    fn unknown_true_hypothesis_is_rejected() {
        let e=observational_equivalence_fixture(); let s=EvaluatorSecret{true_hypothesis_id:"absent".into(),noise_seed:1};
        assert_eq!(execute_reference_probe(&e,&s,"impulse-sensor-x"),Err(ActiveScienceError::UnknownTrueHypothesis));
    }
    #[test]
    fn stale_or_misbound_receipt_is_rejected() {
        let e=observational_equivalence_fixture(); let s=EvaluatorSecret{true_hypothesis_id:"mechanism-basis-x".into(),noise_seed:7};
        let mut r=execute_reference_probe(&e,&s,"impulse-sensor-x").unwrap(); r.episode_id="other".into();
        let p=find_probe(&e,"impulse-sensor-x").unwrap();
        assert_eq!(consistent_hypotheses(&e,p,&r),Err(ActiveScienceError::InvalidReceipt));
    }
    #[test]
    fn corrupted_measurement_removes_every_candidate() {
        let e=observational_equivalence_fixture(); let s=EvaluatorSecret{true_hypothesis_id:"mechanism-basis-x".into(),noise_seed:7};
        let mut r=execute_reference_probe(&e,&s,"impulse-sensor-x").unwrap(); r.observed_trace[1]+=10.0;
        let p=find_probe(&e,"impulse-sensor-x").unwrap();
        assert!(consistent_hypotheses(&e,p,&r).unwrap().is_empty());
    }
    #[test]
    fn observability_is_not_the_same_as_passive_excitation() {
        let e=observational_equivalence_fixture(); let h=&e.hypotheses[0];
        assert_eq!(observability_rank(h,0).unwrap(),2); assert_eq!(observability_rank(h,1).unwrap(),1);
        assert_eq!(predict_trace(&e.hypotheses[0],&e.initial_state,0,&[0.0,0.0]).unwrap(),
                   predict_trace(&e.hypotheses[1],&e.initial_state,0,&[0.0,0.0]).unwrap());
    }
    #[test]
    fn controllability_rank_tracks_input_mechanism() {
        let e=observational_equivalence_fixture();
        assert_eq!(controllability_rank(&e.hypotheses[0]).unwrap(),1);
        assert_eq!(controllability_rank(&e.hypotheses[1]).unwrap(),2);
    }
    #[test]
    fn rejects_state_dimension_outside_two_to_four() {
        let mut e=observational_equivalence_fixture(); e.initial_state=vec![0.0];
        assert_eq!(validate_episode(&e),Err(ActiveScienceError::InvalidEpisode));
    }
    #[test]
    fn duplicate_hypothesis_ids_are_rejected() {
        let mut e=observational_equivalence_fixture(); e.hypotheses[1].hypothesis_id=e.hypotheses[0].hypothesis_id.clone();
        assert_eq!(validate_episode(&e),Err(ActiveScienceError::DuplicateIdentifier));
    }
    #[test]
    fn duplicate_probe_ids_are_rejected() {
        let mut e=observational_equivalence_fixture(); e.permitted_probes[1].probe_id=e.permitted_probes[0].probe_id.clone();
        assert_eq!(validate_episode(&e),Err(ActiveScienceError::DuplicateIdentifier));
    }
    #[test]
    fn tie_breaking_is_stable_under_offer_reordering() {
        let mut e=observational_equivalence_fixture(); e.permitted_probes.swap(1,2);
        assert_eq!(choose_discriminating_probe(&e).unwrap().unwrap().probe_id,"impulse-sensor-x");
    }
    #[test]
    fn nonfinite_probe_inputs_fail_closed() {
        let mut e=observational_equivalence_fixture(); e.permitted_probes[1].input_sequence[0]=f64::NAN;
        assert_eq!(validate_episode(&e),Err(ActiveScienceError::InvalidProbe));
    }
}
