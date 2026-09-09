# NSTDB noise robustness gate

The locked-model robustness harness is implemented in `evaluation.noise_robustness`. It supports the required SNR curve (24, 18, 12, 6, 0, -6 dB) and seeded AWGN smoke tests without fitting or threshold tuning.

The local NSTDB archive is not present yet, so this phase is **not externally validated**. The next data step is to acquire NSTDB 1.0.0 and adapt its noise records to the same PTB-XL/MIT-BIH window contract. No synthetic SNR result should be presented as NSTDB evidence.
