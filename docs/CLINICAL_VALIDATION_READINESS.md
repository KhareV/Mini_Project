# Clinical validation readiness and remaining evidence

## Current claim ceiling

The repository can support **physiological abnormality research monitoring**.
It cannot support diagnosis, treatment decisions, emergency triage, or a
claim of clinical eligibility. Offline public-dataset metrics do not replace a
prospective clinical validation study of the actual AD8232/MAX30102 device.

## Gates already enforced

- Patient-disjoint dataset partitions and ECG duplicate exclusion.
- Validation-only ECG calibration and threshold lock.
- ECG external-domain and noise reports, including negative results.
- Non-circular PPG pulse task with per-subject reporting and quality rejection.
- Explicit rejection of circular and non-generalizing learned fusion targets.
- Hosted input validation, missing/unreliable signal behavior, and component
  provenance in each response.

## Gates requiring hardware or clinical collaborators

- Collect synchronized raw AD8232 ECG and MAX30102 red/IR PPG, not only derived
  HR/SpO2 numbers.
- Compare pulse and SpO2 against an appropriate reference device across normal,
  low-perfusion, motion, and desaturation conditions.
- Freeze MAX30102 ratio-of-ratios calibration coefficients on a calibration
  cohort, then evaluate once on different participants.
- Pre-register endpoints, acceptance limits, exclusions, and sample size.
- Report participant-level confidence intervals and performance by relevant
  demographics/skin pigmentation and clinical state.
- Conduct prospective workflow/usability and clinical-safety review.
- Complete cybersecurity, risk management, data governance, and applicable
  medical-device regulatory review.

No software-only change can truthfully clear these remaining gates.

