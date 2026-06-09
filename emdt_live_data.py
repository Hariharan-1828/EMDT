#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             EMDT — NASA DONKI LIVE SOLAR DATA MODULE                       ║
║                                                                            ║
║   Fetches real solar flare events from NASA's DONKI API.                   ║
║   API: https://kauai.ccmc.gsfc.nasa.gov/DONKI                             ║
║   Authentication: NASA API Key                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import requests
from datetime import datetime, timedelta

# NASA API Key
NASA_API_KEY = 'zvNzXJhAwtNfjhkQQBVUEsbv9cfJw9CYZcU29hrx'


def get_solar_data_live(start_date='2024-05-01', end_date='2024-05-31'):
    """
    Fetch real solar flare data from NASA DONKI API.
    
    Retrieves actual recorded solar flare events and converts them
    to an hourly Kp-index array for the simulation.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    
    Returns
    -------
    t : np.ndarray
        Time axis in hours.
    kp : np.ndarray
        Kp-index values derived from real solar flare data.
    
    Notes
    -----
    Recommended test periods:
    - 2024-05-01 to 2024-05-31: May 2024 geomagnetic superstorm (Kp=9.0)
    - 2023-12-01 to 2023-12-31: Active period, M and X class flares
    - 2024-01-01 to 2024-03-31: Extended 3-month window
    """
    
    # ── Fetch solar flare events from NASA DONKI ─────────────────────────
    url = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR'
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'type': 'ALL',
        'api_key': NASA_API_KEY,
    }
    
    print(f'  🛰️  Fetching NASA DONKI data: {start_date} to {end_date}...')
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        flares = response.json()
        
        if not isinstance(flares, list):
            print(f'  ⚠️  Unexpected API response format. Falling back to synthetic data.')
            from emdt_simulation import get_solar_data
            return get_solar_data()
        
        print(f'  ✅ Found {len(flares)} solar flare events in NASA database')
        
    except requests.exceptions.RequestException as e:
        print(f'  ⚠️  NASA API unavailable: {e}')
        print(f'  📡 Falling back to synthetic data...')
        from emdt_simulation import get_solar_data
        return get_solar_data()
    
    # ── Build hourly time axis for the date range ────────────────────────
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end   = datetime.strptime(end_date,   '%Y-%m-%d')
    n_hours = int((end - start).total_seconds() / 3600)
    
    if n_hours <= 0:
        print(f'  ⚠️  Invalid date range. Falling back to synthetic data.')
        from emdt_simulation import get_solar_data
        return get_solar_data()
    
    t  = np.linspace(0, n_hours, n_hours)
    kp = np.ones(n_hours) * 1.5  # Quiet background
    
    # ── Map each flare event to a Kp spike in the time array ─────────────
    flare_class_to_kp = {
        'X': 8.0,   # Extreme — Kp 8-9
        'M': 5.5,   # Major   — Kp 5-6
        'C': 3.5,   # Common  — Kp 3-4
        'B': 2.0,   # Minor   — Kp 2
        'A': 1.0,   # Quiet   — Kp 1
    }
    
    flares_mapped = 0
    for flare in flares:
        try:
            # Parse flare timestamp
            begin_time = flare.get('beginTime', '')
            if not begin_time:
                continue
            
            flare_time = datetime.strptime(begin_time[:16], '%Y-%m-%dT%H:%M')
            hour_idx = int((flare_time - start).total_seconds() / 3600)
            
            if 0 <= hour_idx < n_hours:
                # Get flare class and map to Kp
                class_type = flare.get('classType', 'B')
                if class_type:
                    cls = class_type[0].upper()
                else:
                    cls = 'B'
                
                peak_kp = flare_class_to_kp.get(cls, 2.0)
                
                # Add Gaussian pulse centered on flare time
                # Width of 4 hours represents typical flare duration
                for offset in range(-12, 12):
                    idx = hour_idx + offset
                    if 0 <= idx < n_hours:
                        kp[idx] += peak_kp * np.exp(-0.5 * (offset / 4) ** 2)
                
                flares_mapped += 1
                
        except (ValueError, KeyError, TypeError):
            continue
    
    # Clamp to real Kp scale: 0 to 9
    kp = np.clip(kp, 0, 9)
    
    print(f'  📊 Flares mapped to timeline: {flares_mapped}/{len(flares)}')
    print(f'  📈 Kp-index built: range {kp.min():.1f} to {kp.max():.1f}')
    print(f'  ⏱️  Time window: {n_hours} hours ({n_hours/24:.0f} days)')
    
    return t, kp


def get_geomagnetic_storms(start_date='2024-05-01', end_date='2024-05-31'):
    """
    Fetch geomagnetic storm events from NASA DONKI.
    
    This provides actual Kp-index values from recorded storms,
    giving the most accurate real-world data.
    
    Returns
    -------
    storms : list
        List of storm event dictionaries from NASA DONKI.
    """
    url = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/GST'
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'api_key': NASA_API_KEY,
    }
    
    print(f'  🌊 Fetching geomagnetic storms: {start_date} to {end_date}...')
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        storms = response.json()
        
        if isinstance(storms, list):
            print(f'  ✅ Found {len(storms)} geomagnetic storm events')
            
            # Print summary of each storm
            for storm in storms:
                storm_id = storm.get('gstID', 'Unknown')
                start_time = storm.get('startTime', 'Unknown')
                
                # Get KP index from allKpIndex if available
                kp_list = storm.get('allKpIndex', [])
                if kp_list:
                    max_kp = max(k.get('kpIndex', 0) for k in kp_list)
                    print(f'        Storm {storm_id}: {start_time[:10]} — Max Kp = {max_kp}')
            
            return storms
        else:
            print(f'  ⚠️  No storm data available')
            return []
            
    except requests.exceptions.RequestException as e:
        print(f'  ⚠️  API error: {e}')
        return []


def get_cme_events(start_date='2024-05-01', end_date='2024-05-31'):
    """
    Fetch Coronal Mass Ejection (CME) events from NASA DONKI.
    
    CMEs are the primary drivers of severe geomagnetic storms.
    This data can be used to enhance the simulation's solar weather model.
    """
    url = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME'
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'api_key': NASA_API_KEY,
    }
    
    print(f'  ☀️  Fetching CME events: {start_date} to {end_date}...')
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        cmes = response.json()
        
        if isinstance(cmes, list):
            print(f'  ✅ Found {len(cmes)} CME events')
            return cmes
        else:
            return []
            
    except requests.exceptions.RequestException as e:
        print(f'  ⚠️  API error: {e}')
        return []


# ── Quick test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 64)
    print('  NASA DONKI API — Connection Test')
    print('=' * 64)
    print()
    
    # Test solar flare data
    t, kp = get_solar_data_live('2024-05-01', '2024-05-31')
    print(f'\n  Result: {len(t)} data points generated')
    print(f'  Kp range: {kp.min():.2f} to {kp.max():.2f}')
    
    # Test geomagnetic storms
    print()
    storms = get_geomagnetic_storms('2024-05-01', '2024-05-31')
    
    # Test CME events
    print()
    cmes = get_cme_events('2024-05-01', '2024-05-31')
    
    print('\n' + '=' * 64)
    print('  ✅ NASA DONKI API test complete')
    print('=' * 64)
