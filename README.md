# EMDT — Electromagnetic Deep-space Transmission System

> **Predictive satellite communication routing with adaptive error correction.**
> Target: IEEE Communications Letters submission.

---

## Overview

EMDT is a research simulation framework for next-generation deep-space communication.
It replaces traditional **reactive** routing (detect failure → wait → reroute) with a **predictive** pipeline that anticipates solar weather disruptions, pre-emptively reroutes traffic, corrects residual bit errors in-situ, and authenticates nodes at the physical layer.

The entire system is validated against **real NASA DONKI solar superstorm data** (May 2024, Kp = 9.0) and produces publication-ready figures, reports, and an IEEE-formatted manuscript.

---

## Project Status — ALL MILESTONES PASSED ✅

| Milestone | Component | IEEE Target | Achieved | Status |
|-----------|-----------|-------------|----------|--------|
| **M1** | AIRC Predictive Routing | > 15 % improvement | **36.2 %** | ✓ PASS |
| **M2** | AEIL Error Recovery (LDPC BP) | BER < 1.1×10⁻⁴ | **4.22×10⁻⁸** | ✓ PASS |
| **M3** | DTN Bundle Protocol Integration | Data continuity | **39.9 % BDR** | ✓ PASS |
| **M4** | RF Eigenfingerprinting Auth | Accuracy > 98 % | **99.3 %** | ✓ PASS |
| **M5** | Full Pipeline Aggregate Gain | > 60 % | **70.8 %** | ✓ PASS |

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- pip

### Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `requests`, `pandas`

### Run the Simulation

```bash
# Default — 30-day synthetic solar data
python run_simulation.py

# Live — NASA DONKI May 2024 superstorm
python run_simulation.py --live --start 2024-05-01 --end 2024-05-31

# Both — side-by-side synthetic vs. live comparison
python run_simulation.py --both

# Custom date range (any period covered by NASA DONKI)
python run_simulation.py --live --start 2023-12-01 --end 2023-12-31

# Custom output directory
python run_simulation.py --output my_results
```

### Run Individual Milestones

```bash
# M3 — Delay Tolerant Networking simulation
python emdt_dtn.py

# M4 — RF Eigenfingerprinting validation
python emdt_rf_fingerprint.py

# Full pipeline — all milestones sequentially (M1–M5)
python master_emdt_pipeline.py
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EMDT PIPELINE                            │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌────────┐ │
│  │ Solar    │───▶│ Channel  │───▶│  AIRC    │──▶│  AEIL  │ │
│  │ Weather  │    │ Model    │    │ Routing  │   │ Decoder │ │
│  │ (Kp)     │    │ (SNR/BER)│    │ (M1)     │   │ (M2)   │ │
│  └──────────┘    └──────────┘    └──────────┘   └────────┘ │
│       │                                │             │      │
│       ▼                                ▼             ▼      │
│  ┌──────────┐                   ┌──────────┐  ┌──────────┐ │
│  │ NASA     │                   │  DTN     │  │  RF Auth │ │
│  │ DONKI API│                   │ BPv7     │  │  SVD     │ │
│  │ (Live)   │                   │ (M3)     │  │  (M4)    │ │
│  └──────────┘                   └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Module | Role |
|--------|------|
| **AIRC** (Adaptive Intelligent Routing Controller) | Looks **15 minutes ahead** using Kp-index forecasts and reroutes before degradation hits. Selects optimal backup path with +1.5 dB advantage. |
| **AEIL** (Adaptive Error Inference Layer) | Rate-1/2 LDPC Belief Propagation decoder delivering **6.5 dB coding gain**, suppressing residual errors down to 10⁻⁸ BER range. |
| **DTN** (Delay Tolerant Networking) | BPv7 store-and-forward bundle protocol handling Mars-Earth orbital occultations and intermittent contact windows. |
| **RF Fingerprinting** | SVD-based hardware eigenfingerprinting that extracts unique phase/amplitude biases to reject rogue spoofing nodes at the physical layer. |

---

## Physics & Signal Models

| Model | Description |
|-------|-------------|
| **ITU-R P.676** | Atmospheric absorption model mapping Kp-index → SNR degradation (up to −14 dB at Kp = 9) |
| **BPSK / AWGN** | Standard modulation + Additive White Gaussian Noise channel: BER = 0.5 × erfc(√SNR) |
| **LDPC Belief Propagation** | Rate-1/2 forward error correction with iterative soft-decision decoding |
| **SVD Feature Extraction** | Singular Value Decomposition on demodulated RF waveforms for hardware fingerprint identification |
| **BPv7 Bundle Protocol** | RFC 9171 interplanetary store-and-forward protocol with custody transfer semantics |

---

## Final Performance Summary

*Validated against NASA DONKI May 2024 Superstorm (Kp = 9.0)*

- **Aggregate Improvement**: **70.8 %** over reactive baselines
- **Delay Reduction**: **70.8 %** in retransmission wait-time (~1100 hours saved)
- **Error Suppression**: Recoverable communication up to **23 dB** into solar flare interference
- **Security**: Hardware-level rejection of **99.3 %** rogue spoofing nodes

---

## Project Structure

```
EMDT/
├── run_simulation.py          # CLI entry point (synthetic / live / both modes)
├── emdt_simulation.py         # Core 8-part simulation engine (M1 + M2)
│                              #   Part 1 — Solar weather generator
│                              #   Part 2 — Channel model (ITU-R P.676, BPSK/AWGN)
│                              #   Part 3 — Reactive router (baseline)
│                              #   Part 4 — EMDT AIRC router (predictive)
│                              #   Part 5 — AEIL decoder (LDPC BP)
│                              #   Part 6 — Performance calculator
│                              #   Part 7 — 8 publication-quality graphs
│                              #   Part 8 — Report printer
├── emdt_live_data.py          # NASA DONKI API integration (solar flares, CMEs, storms)
├── emdt_dtn.py                # M3 — DTN bundle protocol simulation
├── emdt_rf_fingerprint.py     # M4 — RF physical-layer authentication (SVD)
├── master_emdt_pipeline.py    # Orchestrator — runs all milestones sequentially
│
├── ieee_manuscript.tex        # LaTeX manuscript (IEEE Communications Letters format)
├── references.bib             # BibTeX references
├── Final_Master_Scorecard.md  # Consolidated results scorecard
├── requirements.txt           # Python dependencies
│
└── results/                   # Auto-generated output (graphs + reports)
    ├── 01_solar_weather.png   #   Fig 1  — 30-day Kp-index profile
    ├── 02_link_snr.png        #   Fig 2  — Link SNR during solar events
    ├── 03_packet_loss.png     #   Fig 3  — Reactive vs AIRC vs full pipeline
    ├── 04_ber_curves.png      #   Fig 4  — BER: uncoded vs AEIL decoded
    ├── 05_error_suppression.png #  Fig 5  — Error suppression in dB
    ├── 06_delay_comparison.png  #  Fig 6  — Retransmission delay bar chart
    ├── 07_improvements.png    #   Fig 7  — Improvement percentages
    ├── 08_scorecard.png       #   Fig 8  — IEEE target pass/fail scorecard
    ├── simulation_report.txt  #   Text report
    ├── m3/                    #   DTN-specific graphs & reports
    └── m4/                    #   RF fingerprint accuracy curves
```

---

## NASA DONKI API

The project fetches **real solar weather data** from NASA's [DONKI (Space Weather Database)](https://kauai.ccmc.gsfc.nasa.gov/DONKI) including:

- **Solar Flares** (FLR) — mapped to hourly Kp-index via Gaussian pulse injection
- **Geomagnetic Storms** (GST) — actual recorded Kp values
- **Coronal Mass Ejections** (CME) — primary drivers of severe storms

Recommended test periods:
| Period | Notes |
|--------|-------|
| `2024-05-01` to `2024-05-31` | May 2024 geomagnetic superstorm (Kp = 9.0) |
| `2023-12-01` to `2023-12-31` | Active period with M- and X-class flares |
| `2024-01-01` to `2024-03-31` | Extended 3-month observation window |

If the API is unavailable, the simulation gracefully falls back to synthetic data.

---

## Output Graphs

The simulation generates **8 publication-quality figures** (dark-theme, journal-ready) covering:

1. **Solar Weather Input** — Kp-index with flare markers and alert thresholds
2. **Link SNR** — Signal degradation during solar events
3. **Packet Loss Comparison** — Reactive vs AIRC vs full pipeline
4. **BER Curves** — Uncoded BPSK vs AEIL-decoded (log scale)
5. **Error Suppression** — AEIL gain in dB across the SNR range
6. **Retransmission Delay** — Hours saved by EMDT
7. **Improvement Summary** — Horizontal bar chart of all gains
8. **IEEE Scorecard** — Pass/fail status for every milestone target

M3 and M4 produce additional figures (DTN buffer utilization, bundle delivery ratio, RF authentication accuracy vs SNR).

---

## Publication

An IEEE Communications Letters manuscript is included at `ieee_manuscript.tex` with supporting `references.bib`. The paper covers the system architecture, simulation methodology, and all results with embedded figures from the `results/` directory.

---

## License

Research Use / Patent Pending (Provisional)
