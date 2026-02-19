//! Butlin et al. consciousness indicators from 6 neuroscience theories.
//!
//! Tests architectural properties against 14 indicators:
//! RPT (Recurrent Processing), GWT (Global Workspace),
//! HOT (Higher-Order), PP (Predictive Processing),
//! AST (Attention Schema), IIT (Integrated Information).

pub mod indicators;
pub mod report;

pub use indicators::ButlinIndicatorSuite;
pub use report::{ButlinIndicatorReport, IndicatorEvidence, IndicatorStatus};
