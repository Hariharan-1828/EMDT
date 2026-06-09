#!/usr/bin/env python3
"""
EMDT MASTER PIPELINE — FULL PROJECT INTEGRATION
Runs all milestones (M1-M4) and generates the Publication Assets.
"""

import os
import sys
import subprocess

def run_cmd(cmd, desc):
    print(f"\n[MASTER] Running {desc}...")
    try:
        # Use -X utf8 for robustness across environments
        full_cmd = [sys.executable, "-X", "utf8"] + cmd
        subprocess.run(full_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] {desc} failed: {e}")
        return False

def main():
    print("=" * 64)
    print("  EMDT MASTER PIPELINE — VERSION 1.0 (PUBLICATION READY)")
    print("=" * 64)
    
    # 1. Full Pipeline Simulation (M1, M2, M5)
    print("\n--- PHASE 1: Full Simulation & NASA Data (M1, M2, M5) ---")
    run_cmd(["run_simulation.py", "--live"], "Full Simulation")
    
    # 2. M3 DTN Simulation
    print("\n--- PHASE 2: DTN Bundle Protocol (M3) ---")
    run_cmd(["emdt_dtn.py"], "DTN Simulation")
    
    # 3. M4 RF Fingerprinting (M4)
    print("\n--- PHASE 3: RF Authentication Security (M4) ---")
    run_cmd(["emdt_rf_fingerprint.py"], "RF Fingerprinting")
    
    print("\n" + "=" * 64)
    print("  ALL MILESTONES EXECUTED")
    print("=" * 64)
    print("  Results Consolidated in:")
    print("    - results/             (Full Pipeline Graphs)")
    print("    - results/m3/          (DTN Graphs)")
    print("    - results/m4/          (RF Fingerprint Accuracy)")
    print("    - Final_Master_Scorecard.md")
    print("=" * 64)

if __name__ == '__main__':
    main()
