#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 EMDT SIMULATION — RUN ENTRY POINT                          ║
║                                                                            ║
║   Modes:                                                                   ║
║     python run_simulation.py              → Run with synthetic data        ║
║     python run_simulation.py --live       → Run with NASA DONKI live data  ║
║     python run_simulation.py --live --start 2024-05-01 --end 2024-05-31    ║
║     python run_simulation.py --both       → Run synthetic + live side-by-  ║
║                                             side comparison                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import argparse
from emdt_simulation import run_simulation, get_solar_data


def main():
    parser = argparse.ArgumentParser(
        description='EMDT Simulation Engine — Satellite Communication System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py                                    # Synthetic data (default)
  python run_simulation.py --live                             # NASA May 2024 superstorm
  python run_simulation.py --live --start 2024-01-01 --end 2024-03-31
  python run_simulation.py --both                             # Compare synthetic vs live
        """
    )
    
    parser.add_argument('--live', action='store_true',
                        help='Use real NASA DONKI solar data instead of synthetic')
    parser.add_argument('--both', action='store_true',
                        help='Run both synthetic and live data for comparison')
    parser.add_argument('--start', type=str, default='2024-05-01',
                        help='Start date for live data (YYYY-MM-DD). Default: 2024-05-01')
    parser.add_argument('--end', type=str, default='2024-05-31',
                        help='End date for live data (YYYY-MM-DD). Default: 2024-05-31')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for graphs and reports. Default: results')
    
    args = parser.parse_args()
    
    if args.both:
        # ── Run synthetic ────────────────────────────────────────────────
        print('\n' + '=' * 64)
        print('  MODE: SYNTHETIC DATA')
        print('=' * 64 + '\n')
        
        synthetic_dir = os.path.join(args.output, 'synthetic')
        run_simulation(solar_data_func=None, output_dir=synthetic_dir)
        
        # ── Run live ─────────────────────────────────────────────────────
        print('\n\n' + '=' * 64)
        print('  MODE: NASA DONKI LIVE DATA')
        print('=' * 64 + '\n')
        
        from emdt_live_data import get_solar_data_live
        live_func = lambda: get_solar_data_live(args.start, args.end)
        live_dir = os.path.join(args.output, 'live')
        run_simulation(solar_data_func=live_func, output_dir=live_dir)
        
        print('\n' + '=' * 64)
        print(f'  📁 Synthetic results: {synthetic_dir}/')
        print(f'  📁 Live results:      {live_dir}/')
        print('=' * 64)
        
    elif args.live:
        # ── Run with live NASA data ──────────────────────────────────────
        from emdt_live_data import get_solar_data_live
        live_func = lambda: get_solar_data_live(args.start, args.end)
        run_simulation(solar_data_func=live_func, output_dir=args.output)
        
    else:
        # ── Run with synthetic data (default) ────────────────────────────
        run_simulation(solar_data_func=None, output_dir=args.output)


if __name__ == '__main__':
    main()
