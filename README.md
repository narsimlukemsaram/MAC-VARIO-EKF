# MAC-VARIO-EKF

**Metrics-Aware Visual-Inertial Odometry with Uncertainty-Adaptive Error-State EKF Fusion for GPS-Denied Aerial Navigation**


## Overview

MAC-VARIO-EKF investigates visual-inertial state estimation for aerial navigation when sensing quality changes over time. The research framework combines metric-aware visual uncertainty, learned inertial uncertainty, and an error-state extended Kalman filter (EKF) for uncertainty-adaptive fusion. Refer to the paper for the precise state definition, update equations, experimental protocol, and reported results.

> **Release status:** Initial research-code release in preparation. Installation, end-to-end execution, and exact reproduction commands have **not yet been verified from a clean clone**. Do not interpret this repository as a validated, one-command reproduction package until these checks are completed.

## Repository layout

| Path | Purpose |
| --- | --- |
| `MACVIO-main-v3/` | Selected visual-inertial estimator, optimiser, wrappers, preintegration, and trajectory modules |
| `MACI2/mac_i2/` | Visual-inertial initialisation, IMU modelling, covariance and evaluation modules |
| `MACI2/cpu_reproduction/` | Reproduction scripts, sequence configurations and selected evaluation logs |
| `third_party/AirIMU/` | AirIMU upstream dependency (Git submodule) |
| `third_party/MAC-VO/` | MAC-VO upstream dependency (Git submodule) |


## Obtain the code

```bash
git clone --recurse-submodules https://github.com/<OWNER>/<REPOSITORY>.git
cd <REPOSITORY>
```

Replace `<OWNER>/<REPOSITORY>` with the published GitHub location. The MAC-VO submodule has its own nested dependencies; recursive cloning is intended to retrieve them, subject to upstream accessibility.

The checked-out upstream revisions for this prepared release are:

- AirIMU: `119fe7ce229e321b3890282909a183aefad61dfb`
- MAC-VO: `8683b532d6f6e8981f6b1a646fb5f52f02c1ea61`

## Setup and evaluation

The implementation uses Python and third-party packages, including PyTorch and packages required by MAC-VO and AirIMU. **A verified top-level environment specification and end-to-end run command are still to be supplied.** Consult the upstream projects' documentation and the relevant configuration files under `MACI2/cpu_reproduction/configs/` in the meantime. Do not assume that the available evaluation logs alone establish reproducibility.

## Data and model weights

EuRoC data, large frontend caches, and pretrained checkpoints are **not included**. Obtain datasets from their original maintainers and confirm the applicable terms of use. Any training or inference checkpoints must be obtained separately, with their original provenance and permissions verified. Some local configurations may contain machine-specific paths and will require editing before use on another system.

## Attribution

This code builds on work from:

- [MAC-VO](https://github.com/MAC-VO/MAC-VO), distributed under its upstream MIT licence.
- [AirIMU](https://github.com/haleqiu/AirIMU), distributed under its upstream BSD 3-Clause licence.

Keep all upstream copyright and licence notices intact. The licences of nested dependencies and model weights must be reviewed separately. The licence for original MAC-VARIO-EKF contributions has not yet been selected.

## Citation

We will add a verified public bibliographic record and citation entry when available. 
