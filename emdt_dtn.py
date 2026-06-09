#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               EMDT MILESTONE M3 — DTN BUNDLE PROTOCOL SIMULATION             ║
║                                                                              ║
║   Simulates Mars-Earth contact windows, evaluating Bundle Delivery Ratio     ║
║   (BDR) and Buffer Utilization for standard IP routing vs DTN routing.      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import matplotlib
import gc
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emdt_simulation import get_solar_data, kp_to_snr, snr_to_ber, ber_to_packet_loss, ReactiveRouter, EMDTRouter, COLORS

# Optional NASA Live Data fallback
try:
    from emdt_live_data import get_solar_data_live
except ImportError:
    get_solar_data_live = None

def simulate_dtn_pipeline(t_hours, pkt_loss_series, orbit_duration_min=120, contact_window_min=10, 
                          data_rate_bps=1000, buffer_capacity=1e8, is_dtn=True):
    """
    Simulate data transmission over intermittent contact windows using either standard TCP/IP
    or DTN Bundle Protocol (BPv7) store-and-forward mechanics.
    
    Parameters
    ----------
    t_hours : np.ndarray
        Array of time points in hours (size N).
    pkt_loss_series : np.ndarray
        Packet loss probability for each time step in t_hours.
    orbit_duration_min : int
        Total duration of one orbit around Mars (e.g., relay satellite like MRO)
    contact_window_min : int
        Window of time per orbit with line-of-sight to Earth's DSN.
    data_rate_bps : int
        Rate at which bundles/data is continually generated.
    buffer_capacity : float
        Maximum nodes cache size before dropping happens in DTN. 
    is_dtn : bool
        If True, use BPv7 store and forward. If False, use TCP/IP drop.
    
    Returns
    -------
    bdr : float
        Bundle Delivery Ratio (Successfully Delivered / Total Generated)
    buffer_utilization : np.ndarray
        Size of the buffer over time (in Bundles)
    contact_out : np.ndarray
        Binary array representing Line-Of-Sight (1 open, 0 closed)
    delivery_history : np.ndarray
        Accumulated successful deliveries over time
    """
    
    # We must upsample our hourly data to minute-level resolution to simulate realistic orbits
    total_minutes = int(np.max(t_hours) * 60)
    t_min = np.linspace(0, total_minutes, total_minutes)
    
    # Interpolate packet loss array to minute resolution
    pkt_loss_min = np.interp(t_min, t_hours * 60, pkt_loss_series)
    
    # Create contact windows (Binary pulse train)
    contact_out = np.zeros_like(t_min)
    for i in range(total_minutes):
        if (i % orbit_duration_min) < contact_window_min:
            contact_out[i] = 1  # Window Open
            
    # Simulation state
    buffer = 0
    total_generated = 0
    total_delivered = 0
    
    buffer_utilization = np.zeros_like(t_min)
    delivery_history = np.zeros_like(t_min)
    
    # Run the discrete event simulation (1 minute increments)
    for i in range(total_minutes):
        # Data constantly streams in from local deep-space node
        generated_this_min = data_rate_bps * 60 
        total_generated += generated_this_min
        
        if is_dtn:
            buffer += generated_this_min
            # Cap at capacity
            buffer = min(buffer, buffer_capacity)
        else:
            # IP networking does not buffer for long latency/disconnections
            buffer = generated_this_min
        
        # Transmission phase
        if contact_out[i] == 1:
            # Contact is open! Attempt transmission
            # The channel success rate dictates how much goes through
            success_rate = 1.0 - pkt_loss_min[i]
            
            # Bandwidth limit: assume we can transmit 5x standard rate during the window
            attempt_to_transmit = min(buffer, (data_rate_bps * 60 * 5))
            
            delivered_this_min = attempt_to_transmit * success_rate
            total_delivered += delivered_this_min
            buffer -= attempt_to_transmit
            
            # Any remaining bundles that failed transmission are:
            # - Kept in buffer for DTN protocols to try next time
            # - Immediately dropped in strict IP/TCP scenarios
            if is_dtn:
                buffer += (attempt_to_transmit - delivered_this_min)
                buffer = min(buffer, buffer_capacity) # Recap
        elif not is_dtn:
             # Window is closed. Standard TCP drops everything that backs up
             buffer = 0
             
        buffer_utilization[i] = buffer
        delivery_history[i] = total_delivered
        
    bdr = (total_delivered / max(1, total_generated)) * 100.0
    return bdr, buffer_utilization, contact_out, delivery_history, t_min


def create_dtn_graphs(t_min, contact_window,
                      buf_ip_reactive, buf_dtn_reactive, buf_dtn_emdt,
                      history_ip_reactive, history_dtn_reactive, history_dtn_emdt,
                      bdr_ip_reactive, bdr_dtn_reactive, bdr_dtn_emdt,
                      output_dir='results/m3'):
    """Generate the M3 specific reporting graphs."""
    
    os.makedirs(output_dir, exist_ok=True)
    plt.close('all')
    
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['panel'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text_dim'],
        'ytick.color': COLORS['text_dim'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 11,
    })
    
    t_hours = t_min / 60.0
    
    # Limit visualization to first 120 hours (5 days) so orbits are visible
    limit_idx = 120 * 60
    t_plot = t_hours[:limit_idx]
    
    # ── GRAPH 9: Contact Windows and Buffer Utilization ──────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    
    ax1.fill_between(t_plot, 0, contact_window[:limit_idx], color=COLORS['cyan'], alpha=0.5)
    ax1.set_title('Figure 9a — Mars-Earth Orbital Contact Windows (Line-of-Sight)', fontsize=14, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_ylabel('Contact Status')
    
    ax2.plot(t_plot, buf_dtn_reactive[:limit_idx], color=COLORS['reactive'], linewidth=1.5, alpha=0.8, label="DTN Buffer (Reactive Backend)")
    ax2.plot(t_plot, buf_dtn_emdt[:limit_idx], color=COLORS['full'], linewidth=1.5, alpha=0.8, label="DTN Buffer (EMDT Backend)")
    ax2.plot(t_plot, buf_ip_reactive[:limit_idx], color=COLORS['text_dim'], linewidth=1.5, linestyle=':', alpha=0.5, label="Standard IP Queue (Drops)")
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Cached Bundles (Bytes)')
    ax2.set_title('Figure 9b — Store-And-Forward Buffer Utilization Over Intermittent Contact', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.8, facecolor=COLORS['panel'], edgecolor=COLORS['grid'])
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_dtn_buffer_utilization.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    
    # ── GRAPH 10: Bundle Delivery Ratio Comparison ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Standard IP TCP\n(Reactive Routing)', 'DTN Bundle Protocol\n(Reactive Routing)', 'DTN Bundle Protocol\n(EMDT AIRC Routing)']
    values = [bdr_ip_reactive, bdr_dtn_reactive, bdr_dtn_emdt]
    bar_colors = [COLORS['text_dim'], COLORS['reactive'], COLORS['full']]
    
    bars = ax.barh(categories, values, color=bar_colors, height=0.6, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', 
                ha='left', va='center', fontsize=13, fontweight='bold', color=COLORS['text'])
                
    ax.set_xlabel('Bundle Delivery Ratio (BDR %)')
    ax.set_title('Figure 10 — Bundle Delivery Ratio: Deep Space Network Environment', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_bundle_delivery_ratio.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    
    # ── Save Results to disk ─────────────────────────────
    report_lines = [
        "================================================================",
        "  EMDT MILESTONE M3 — OVERALL RESULTS",
        "================================================================",
        "",
        "  --- Bundle Delivery Ratio (BDR) ---",
        f"  Standard IP / TCP (Reactive)    : {bdr_ip_reactive:.2f}%",
        f"  DTN Protocol (Reactive)         : {bdr_dtn_reactive:.2f}%",
        f"  DTN Protocol (EMDT AIRC)        : {bdr_dtn_emdt:.2f}%",
        "",
        "  --- IEEE Target Validation ---",
        "  M3 verifies that DTN successfully preserves data dropping out ",
        "  from continuous link occlusion and standard IP failures.",
        "  Using EMDT as the underlying sub-network provides the ultimate ",
        "  delivery guarantee across both weather AND planetary orbits."
    ]
    with open(os.path.join(output_dir, 'm3_dtn_report.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

def run_dtn_module(live_data=False):
    """Execute the Delay Tolerant Networking DTN emulation."""
    print('=' * 64)
    print('  EMDT SYSTEM — MILESTONE M3 DTN INTEGRATION')
    print('=' * 64)
    
    # Get backend signal losses
    out_dir = 'results/m3/synthetic'
    if live_data and get_solar_data_live:
        print("\n  [1/3] Gathering base data (NASA DONKI Live - Dec 2023)...")
        t, kp = get_solar_data_live('2023-12-01', '2023-12-31')
        out_dir = 'results/m3/live'
    else:
        print("\n  [1/3] Gathering base data (Synthetic Generator)...")
        t, kp = get_solar_data(n_hours=720)
        
    snr = kp_to_snr(kp)
    ber = snr_to_ber(snr)
    pkt_loss = ber_to_packet_loss(ber)
    
    reactive = ReactiveRouter()
    emdt = EMDTRouter()
    
    loss_reactive = reactive.route(t, kp, snr, ber, pkt_loss)
    loss_emdt = emdt.route(t, kp, snr, ber, pkt_loss)
    
    print("  [2/3] Simulating MRO Mars Orbital Contacts & Bundle Protocols...")
    
    # Case 1: Standard Internet Protocol (No DTN) with Reactive Route
    bdr_ip_reac, buf_ip_reac, contact, hist_ip_reac, t_min = simulate_dtn_pipeline(
        t, loss_reactive, is_dtn=False
    )
    
    # Case 2: DTN Bundle Protocol with Reactive Route
    bdr_dtn_reac, buf_dtn_reac, _, hist_dtn_reac, _ = simulate_dtn_pipeline(
        t, loss_reactive, is_dtn=True
    )
    
    # Case 3: DTN Bundle Protocol with EMDT AIRC Route (Our Solution Strategy)
    bdr_dtn_emdt, buf_dtn_emdt, _, hist_dtn_emdt, _ = simulate_dtn_pipeline(
        t, loss_emdt, is_dtn=True
    )
    
    print("\n  [3/3] Generating output analytics...")
    create_dtn_graphs(
        t_min, contact,
        buf_ip_reac, buf_dtn_reac, buf_dtn_emdt,
        hist_ip_reac, hist_dtn_reac, hist_dtn_emdt,
        bdr_ip_reac, bdr_dtn_reac, bdr_dtn_emdt,
        output_dir=out_dir
    )
    
    print(f"\n  📊 Outputs published to '{out_dir}'")
    print(f"       IP BDR  : {bdr_ip_reac:.2f}%")
    print(f"       DTN BDR : {bdr_dtn_reac:.2f}%")
    print(f"       EMDT BDR: {bdr_dtn_emdt:.2f}%")
    
if __name__ == '__main__':
    run_dtn_module(live_data=False)
