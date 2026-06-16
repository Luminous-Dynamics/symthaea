use symthaea_core::autopoietic::mod::SymthaeaAutopoieticOrchestrator;
use symthaea_core::hdc::morphogenetic_bridge::MorphogeneticState;

struct MockSubsystem { health: f32, energy: f32 }
impl MorphogeneticState for MockSubsystem {
    fn energy(&self) -> f32 { self.energy }
    fn health(&self) -> f32 { self.health }
    fn self_repair(&mut self) -> usize { self.health += 0.1; 1 }
    fn metabolic_budget(&self) -> f32 { self.health }
}

#[test]
fn test_orchestrator_stress_response() {
    let mut orchestrator = SymthaeaAutopoieticOrchestrator::new(100.0);
    
    // System under stress (low health)
    orchestrator.register_subsystem(Box::new(MockSubsystem { health: 0.1, energy: 0.9 }));
    orchestrator.register_subsystem(Box::new(MockSubsystem { health: 0.2, energy: 0.8 }));
    
    // Run maintenance cycle
    let avg_health_start = orchestrator.maintenance_cycle();
    let avg_health_end = orchestrator.maintenance_cycle();
    
    assert!(avg_health_end > avg_health_start, "Orchestrator should improve system health under stress");
}
