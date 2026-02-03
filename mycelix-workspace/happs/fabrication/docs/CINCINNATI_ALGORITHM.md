# Cincinnati Algorithm

> Teleomorphic monitoring for real-time print quality assurance

The Cincinnati Algorithm provides real-time quality monitoring during 3D printing, enabling early anomaly detection, adaptive parameter adjustment, and comprehensive quality records.

## Overview

Named after the pioneering work in acoustic emission monitoring at the University of Cincinnati, this algorithm extends those principles to multi-sensor telemetry for distributed manufacturing quality assurance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CINCINNATI MONITORING FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SENSORS            PROCESSING         DECISION          ACTION           │
│                                                                             │
│   ┌──────────┐      ┌───────────┐      ┌──────────┐      ┌──────────┐     │
│   │ Temp     │─────▶│           │      │          │      │ Continue │     │
│   │ Hotend   │      │  Feature  │      │ Anomaly  │──────▶│   or     │     │
│   ├──────────┤      │Extraction │─────▶│Detection │      │ Adjust   │     │
│   │ Temp Bed │─────▶│           │      │          │──────▶│   or     │     │
│   ├──────────┤      │  + Local  │      │          │      │ Pause    │     │
│   │ Stepper  │─────▶│  Baseline │      │          │──────▶│   or     │     │
│   │ Currents │      │  Compare  │      │          │      │ Abort    │     │
│   ├──────────┤      │           │      │          │      └──────────┘     │
│   │ Vibration│─────▶│           │      │          │             │          │
│   │ RMS      │      └───────────┘      └──────────┘             │          │
│   ├──────────┤                                                   ▼          │
│   │ Filament │                              ┌─────────────────────────┐     │
│   │ Tension  │                              │     Quality Report      │     │
│   └──────────┘                              │  - Layer scores         │     │
│                                             │  - Anomaly events       │     │
│                                             │  - Overall health       │     │
│                                             └─────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Sensor Requirements

### Required Sensors

| Sensor | Sampling Rate | Purpose |
|--------|---------------|---------|
| Hotend Temperature | 10 Hz | Extrusion consistency |
| Bed Temperature | 1 Hz | Adhesion monitoring |
| Stepper Currents (X/Y/Z/E) | 1000 Hz | Mechanical anomalies |
| Vibration (IMU) | 1000 Hz | Layer adhesion, resonance |

### Optional Sensors

| Sensor | Sampling Rate | Purpose |
|--------|---------------|---------|
| Filament Tension | 100 Hz | Feed issues |
| Ambient Temperature | 0.1 Hz | Environmental factors |
| Humidity | 0.1 Hz | Material absorption |
| Camera | 1 Hz | Visual anomaly detection |

### Sensor Snapshot

```typescript
interface SensorSnapshot {
  hotendTemp: number;                      // Celsius
  bedTemp: number;                         // Celsius
  stepperCurrents: [number, number, number, number]; // mA [X, Y, Z, E]
  vibrationRms: number;                    // g
  filamentTension?: number;                // grams
  ambientTemp?: number;                    // Celsius
  humidity?: number;                       // %RH
}
```

## Baseline Establishment

### Initial Calibration

Before monitoring begins, establish a "healthy print" baseline:

```typescript
interface BaselineSignature {
  // Temperature profiles per layer height
  tempProfiles: Map<number, TempProfile>;

  // Stepper current patterns for common moves
  movePatterns: {
    infill: CurrentPattern;
    perimeter: CurrentPattern;
    travel: CurrentPattern;
    retract: CurrentPattern;
  };

  // Vibration signature at different speeds
  vibrationProfile: Map<number, VibrationPattern>;

  // Layer-specific expectations
  layerExpectations: LayerExpectation[];
}
```

### Learning Period

First 5-10 prints with a new design establish baseline:

```typescript
async function establishBaseline(
  designHash: string,
  settings: PrintSettings
): Promise<BaselineSignature> {
  const samples: SensorSnapshot[][] = [];

  // Collect from multiple successful prints
  for (const printRecord of getSuccessfulPrints(designHash, 5)) {
    samples.push(await getSessionTelemetry(printRecord.cincinnatiSessionId));
  }

  // Statistical analysis
  return {
    tempProfiles: calculateTempProfiles(samples),
    movePatterns: extractMovePatterns(samples),
    vibrationProfile: buildVibrationProfile(samples),
    layerExpectations: generateLayerExpectations(samples),
  };
}
```

## Anomaly Detection

### Anomaly Types

```typescript
type AnomalyType =
  | 'ExtrusionInconsistency'  // Temperature/current deviation during extrusion
  | 'TemperatureDeviation'    // Hotend or bed temp outside bounds
  | 'VibrationAnomaly'        // Unexpected vibration patterns
  | 'LayerAdhesionFailure'    // Z-axis anomaly suggesting delamination
  | 'NozzleClog'              // High extruder current, low flow
  | 'BedLevelDrift'           // Z-compensation changes over time
  | 'PowerFluctuation'        // Supply voltage/current issues
  | 'Unknown';                // Unclassified anomaly
```

### Detection Algorithms

#### Extrusion Consistency

```typescript
function detectExtrusionAnomaly(
  snapshot: SensorSnapshot,
  baseline: BaselineSignature,
  currentMove: MoveType
): AnomalyEvent | null {
  const expectedCurrent = baseline.movePatterns[currentMove].extruderCurrent;
  const deviation = Math.abs(snapshot.stepperCurrents[3] - expectedCurrent.mean);

  if (deviation > expectedCurrent.stdDev * 3) {
    return {
      anomalyType: 'ExtrusionInconsistency',
      severity: Math.min(1.0, deviation / (expectedCurrent.stdDev * 5)),
      sensorData: snapshot,
    };
  }
  return null;
}
```

#### Temperature Monitoring

```typescript
function detectTemperatureAnomaly(
  snapshot: SensorSnapshot,
  targetTemp: number,
  threshold: number = 5
): AnomalyEvent | null {
  const deviation = Math.abs(snapshot.hotendTemp - targetTemp);

  if (deviation > threshold) {
    return {
      anomalyType: 'TemperatureDeviation',
      severity: Math.min(1.0, deviation / 20),
      sensorData: snapshot,
    };
  }
  return null;
}
```

#### Vibration Analysis

```typescript
function detectVibrationAnomaly(
  snapshot: SensorSnapshot,
  baseline: BaselineSignature,
  printSpeed: number
): AnomalyEvent | null {
  const expectedRms = baseline.vibrationProfile.get(printSpeed)?.rms || 0.5;
  const deviation = Math.abs(snapshot.vibrationRms - expectedRms);

  // High vibration could indicate:
  // - Loose belt
  // - Layer adhesion failure
  // - Resonance issues
  if (deviation > expectedRms * 0.5) {
    return {
      anomalyType: 'VibrationAnomaly',
      severity: Math.min(1.0, deviation / expectedRms),
      sensorData: snapshot,
    };
  }
  return null;
}
```

#### Nozzle Clog Detection

```typescript
function detectNozzleClog(
  snapshot: SensorSnapshot,
  baseline: BaselineSignature
): AnomalyEvent | null {
  const extruderCurrent = snapshot.stepperCurrents[3];
  const expectedCurrent = baseline.movePatterns.infill.extruderCurrent.mean;

  // Clog: High current (motor straining) but no movement
  // Or: Normal current but filament tension spike
  const currentRatio = extruderCurrent / expectedCurrent;
  const tensionSpike = snapshot.filamentTension
    ? snapshot.filamentTension > 500  // grams
    : false;

  if (currentRatio > 1.5 || tensionSpike) {
    return {
      anomalyType: 'NozzleClog',
      severity: Math.min(1.0, (currentRatio - 1) / 2),
      sensorData: snapshot,
    };
  }
  return null;
}
```

### Anomaly Event Structure

```typescript
interface AnomalyEvent {
  timestampMs: number;           // Since print start
  layerNumber: number;           // Current layer
  anomalyType: AnomalyType;
  severity: number;              // 0.0-1.0
  sensorData: SensorSnapshot;    // Raw data at anomaly
  actionTaken?: CincinnatiAction; // What was done
}
```

## Adaptive Response

### Action Types

```typescript
type CincinnatiAction =
  | 'Continue'                                    // All normal
  | { AdjustParameters: Record<string, number> }  // Auto-tune
  | 'PauseForInspection'                          // Human review needed
  | { AbortPrint: string }                        // Fatal issue
  | { AlertOperator: string };                    // Notification only
```

### Decision Logic

```typescript
function decideCincinnatiAction(
  anomaly: AnomalyEvent,
  safetyClass: SafetyClass,
  history: AnomalyEvent[]
): CincinnatiAction {
  // Count recent anomalies of same type
  const recentSameType = history.filter(
    h => h.anomalyType === anomaly.anomalyType &&
         h.timestampMs > anomaly.timestampMs - 60000
  ).length;

  // Critical safety classes are more conservative
  const severityThreshold = safetyClass >= 'Class3BodyContact' ? 0.3 : 0.5;
  const countThreshold = safetyClass >= 'Class3BodyContact' ? 2 : 5;

  // High severity immediate abort
  if (anomaly.severity > 0.9) {
    return { AbortPrint: `Critical ${anomaly.anomalyType}` };
  }

  // Repeated issues escalate
  if (recentSameType >= countThreshold) {
    return 'PauseForInspection';
  }

  // Try auto-adjustment for some anomaly types
  if (anomaly.anomalyType === 'ExtrusionInconsistency' && anomaly.severity < 0.5) {
    return {
      AdjustParameters: {
        flowRate: 0.95, // Reduce flow slightly
      },
    };
  }

  if (anomaly.anomalyType === 'TemperatureDeviation' && anomaly.severity < 0.4) {
    return {
      AdjustParameters: {
        printSpeed: 0.9, // Slow down to stabilize
      },
    };
  }

  // Low severity, continue with alert
  if (anomaly.severity < severityThreshold) {
    return { AlertOperator: `Minor ${anomaly.anomalyType} detected` };
  }

  // Default to pause for inspection
  return 'PauseForInspection';
}
```

## Quality Scoring

### Layer-by-Layer Scoring

```typescript
function scoreLayer(
  layerNumber: number,
  samples: SensorSnapshot[],
  baseline: BaselineSignature,
  anomalies: AnomalyEvent[]
): number {
  // Base score from sensor consistency
  let score = 1.0;

  // Temperature stability
  const tempVariance = calculateVariance(samples.map(s => s.hotendTemp));
  const expectedVariance = 2.0; // Celsius
  score -= Math.min(0.2, (tempVariance / expectedVariance) * 0.1);

  // Vibration consistency
  const vibrationVariance = calculateVariance(samples.map(s => s.vibrationRms));
  score -= Math.min(0.2, vibrationVariance * 0.5);

  // Anomaly penalties
  const layerAnomalies = anomalies.filter(a => a.layerNumber === layerNumber);
  for (const anomaly of layerAnomalies) {
    score -= anomaly.severity * 0.3;
  }

  return Math.max(0, Math.min(1, score));
}
```

### Overall Health Score

```typescript
function calculateOverallHealth(
  layerScores: number[],
  anomalies: AnomalyEvent[]
): number {
  // Average layer scores
  const avgLayerScore = layerScores.reduce((a, b) => a + b, 0) / layerScores.length;

  // Anomaly penalty (diminishing returns)
  const anomalyPenalty = Math.min(0.3, anomalies.length * 0.02);

  // Worst layer penalty (catch catastrophic single layers)
  const minScore = Math.min(...layerScores);
  const worstLayerPenalty = (1 - minScore) * 0.2;

  return Math.max(0, avgLayerScore - anomalyPenalty - worstLayerPenalty);
}
```

## Report Generation

### Cincinnati Report Structure

```typescript
interface CincinnatiReport {
  sessionId: string;
  totalSamples: number;          // Total sensor readings
  anomaliesDetected: number;     // Count of anomaly events
  anomalyEvents: AnomalyEvent[]; // Full event list
  overallHealthScore: number;    // 0.0-1.0
  layerByLayerScores: number[];  // Score per layer
  recommendedAction: CincinnatiAction; // Post-print recommendation
}
```

### Report Generation

```typescript
async function generateCincinnatiReport(
  session: CincinnatiSession
): Promise<CincinnatiReport> {
  const telemetry = await getSessionTelemetry(session.sessionId);
  const anomalies = session.anomalyEvents;

  // Calculate layer scores
  const layerScores: number[] = [];
  for (let layer = 0; layer < telemetry.totalLayers; layer++) {
    const layerSamples = telemetry.getSamplesForLayer(layer);
    layerScores.push(scoreLayer(layer, layerSamples, session.baseline, anomalies));
  }

  const overallHealth = calculateOverallHealth(layerScores, anomalies);

  // Determine post-print recommendation
  let recommendedAction: CincinnatiAction = 'Continue';
  if (overallHealth < 0.5) {
    recommendedAction = { AlertOperator: 'Print quality below threshold' };
  } else if (overallHealth < 0.7 && anomalies.length > 10) {
    recommendedAction = { AlertOperator: 'Multiple anomalies detected - inspect before use' };
  }

  return {
    sessionId: session.sessionId,
    totalSamples: telemetry.totalSamples,
    anomaliesDetected: anomalies.length,
    anomalyEvents: anomalies,
    overallHealthScore: overallHealth,
    layerByLayerScores: layerScores,
    recommendedAction,
  };
}
```

## Safety Class Requirements

### Class 0-2: Optional Monitoring

Monitoring is optional but recommended:
- Basic temperature logging
- Manual quality assessment sufficient

### Class 3+: Required Monitoring

Full Cincinnati monitoring mandatory:

| Requirement | Class 3 | Class 4 | Class 5 |
|-------------|---------|---------|---------|
| All required sensors | Yes | Yes | Yes |
| Min sampling rate | 100 Hz | 500 Hz | 1000 Hz |
| Max anomaly count | 20 | 10 | 5 |
| Min health score | 0.7 | 0.8 | 0.9 |
| Human inspection | Optional | Required | Required + Sign-off |
| Report retention | 90 days | 1 year | Permanent |

## Integration

### Starting a Monitoring Session

```typescript
async function startCincinnatiSession(
  printJobId: string,
  printerCapabilities: PrinterCapabilities
): Promise<CincinnatiSession> {
  // Determine sampling rate based on safety class
  const design = await getDesignForJob(printJobId);
  const samplingRate = getSamplingRateForSafetyClass(design.safetyClass);

  // Load or establish baseline
  const baseline = await getOrEstablishBaseline(
    design.id,
    printJobId.settings
  );

  return {
    sessionId: generateSessionId(),
    estimatorVersion: CINCINNATI_VERSION,
    samplingRate,
    baselineSignature: baseline.signature,
    active: true,
    startedAt: Date.now(),
    anomalyEvents: [],
  };
}
```

### Real-time Monitoring Loop

```typescript
async function monitoringLoop(
  session: CincinnatiSession,
  sensorStream: AsyncIterable<SensorSnapshot>
): Promise<void> {
  const anomalies: AnomalyEvent[] = [];
  let currentLayer = 0;

  for await (const snapshot of sensorStream) {
    // Update layer tracking
    currentLayer = await getCurrentLayer();

    // Run all anomaly detectors
    const detectedAnomalies = [
      detectExtrusionAnomaly(snapshot, session.baseline, getCurrentMove()),
      detectTemperatureAnomaly(snapshot, session.targetTemp),
      detectVibrationAnomaly(snapshot, session.baseline, session.printSpeed),
      detectNozzleClog(snapshot, session.baseline),
    ].filter(a => a !== null);

    for (const anomaly of detectedAnomalies) {
      anomaly.timestampMs = Date.now() - session.startedAt;
      anomaly.layerNumber = currentLayer;

      // Decide action
      const action = decideCincinnatiAction(
        anomaly,
        session.safetyClass,
        anomalies
      );
      anomaly.actionTaken = action;

      // Execute action
      await executeAction(action, session.printJobId);

      // Record anomaly
      anomalies.push(anomaly);

      // Check for abort
      if ('AbortPrint' in action) {
        session.active = false;
        break;
      }
    }

    // Store samples for layer scoring
    await storeSample(session.sessionId, snapshot, currentLayer);
  }

  session.anomalyEvents = anomalies;
}
```

### Completing a Session

```typescript
async function completeCincinnatiSession(
  session: CincinnatiSession
): Promise<CincinnatiReport> {
  session.active = false;

  const report = await generateCincinnatiReport(session);

  // Store report on DHT
  await storeReport(report);

  // Update baseline with successful print data
  if (report.overallHealthScore > 0.8 && report.anomaliesDetected < 5) {
    await updateBaseline(session.designHash, session.sessionId);
  }

  return report;
}
```

## Future Extensions

### Machine Learning Integration

```typescript
// Train anomaly detection model on historical data
async function trainAnomalyModel(
  designHash: string
): Promise<AnomalyModel> {
  const historicalSessions = await getHistoricalSessions(designHash);
  const labeledData = await getLabeledAnomalies(historicalSessions);

  return trainModel(labeledData, {
    type: 'RandomForest',
    features: ['temp', 'current', 'vibration', 'tension'],
  });
}
```

### Visual Inspection Integration

```typescript
// Camera-based layer inspection
async function visualLayerInspection(
  layerImage: ImageData
): Promise<VisualAnomaly[]> {
  const model = await loadVisualModel();
  return model.detectAnomalies(layerImage);
}
```

### Federated Learning

Share anonymized patterns across the network:

```typescript
// Contribute patterns without revealing design details
async function contributeToFederatedLearning(
  session: CincinnatiSession
): Promise<void> {
  const anonymizedPatterns = anonymizePatterns(session);
  await submitToFLCoordinator(anonymizedPatterns);
}
```

---

*Quality is not an act, it is a habit. Cincinnati makes that habit automatic.*
