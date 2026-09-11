# Reviewer Demo V1

Open `/demo` from the dashboard's **DEMO / Simulation** navigation item.
The scenario is deterministic and runs entirely in the browser, so it does not
require the backend, database, simulator service, or physical sensors.

## Scripted progression

1. Stable baseline — reliable ECG/PPG, normal pulse and SpO2.
2. Activity transition — gradual pulse increase without an alert.
3. Motion artefact — quality falls below 0.50 and prediction is suppressed as
   `UNRELIABLE SIGNAL` despite an elevated raw risk score.
4. Abnormal vitals — signal quality recovers, pulse exceeds 100 BPM, measured
   SpO2 falls below 95%, and the event state opens.
5. Recovery — readings normalize and the event progresses toward clear.

## Presentation controls

- Play/pause and reset.
- Jump directly to the next phase or select any phase on the timeline.
- 0.5x, 1x, 2x, and 4x playback speeds.
- Animated ECG and PPG waveforms tied to simulated pulse and quality.
- Live component values, model score, quality bars, event state, explanation,
  and progression log.

Every surface is marked as simulated and carries a research-only disclaimer.
The display demonstrates software behavior, not clinical model performance.
