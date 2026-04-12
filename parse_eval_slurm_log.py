#!/usr/bin/env python3
"""
Parse comparison log file to extract MIP vs Metaheuristic results
Recovers data from incomplete runs that hit time limits
"""

import re
import pandas as pd
import sys
from pathlib import Path


def parse_comparison_log(log_file_path, area=None, hw=None, mh_technique=None):
    """
    Parse log file to extract comparison results between MIP and metaheuristic solutions
    
    Args:
        log_file_path: Path to the log file
        area: Area constraint (extracted from log if not provided)
        hw: Hardware scale factor (extracted from log if not provided)
        mh_technique: Metaheuristic technique name (extracted from log if not provided)
        
    Returns:
        pd.DataFrame with extracted results
    """
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Pattern to match the start of each seed/strategy combo
    seed_strategy_pattern = r'working on seed (\d+) strategy (\w+)'
    
    # Pattern to match the final comparison output
    comparison_pattern = (
        r'optimal solution has a makespan of ([\d.]+), uses ([\d.]+) of total area\s+'
        r'metaheuristic solution has a makespan of ([\d.]+), uses ([\d.]+) of total area'
    )
    
    results = []
    
    # Split log into chunks for each seed/strategy combination
    chunks = re.split(seed_strategy_pattern, log_content)
    
    # chunks[0] is content before first match, then alternates (seed, strategy, content)
    for i in range(1, len(chunks), 3):
        if i + 2 >= len(chunks):
            break
            
        seed = int(chunks[i])
        strategy = chunks[i + 1]
        content = chunks[i + 2]
        
        # Try to find the comparison output
        match = re.search(comparison_pattern, content)
        
        if match:
            mip_cost = float(match.group(1))
            mip_area_used = float(match.group(2))
            mh_cost = float(match.group(3))
            mh_area_used = float(match.group(4))
            
            result = {
                'seed': seed,
                'strategy': strategy,
                'mip_cost': mip_cost,
                'mh_cost': mh_cost,
                'mip_area_used': mip_area_used,
                'mh_area_used': mh_area_used
            }
            
            results.append(result)
            print(f"✓ Extracted: seed={seed}, strategy={strategy}, mip={mip_cost:.2f}, mh={mh_cost:.2f}")
        else:
            print(f"✗ Incomplete: seed={seed}, strategy={strategy} (no comparison output found)")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add config parameters if provided
    if area is not None:
        df['area'] = area
    if hw is not None:
        df['hw'] = hw
    if mh_technique is not None:
        df['mh'] = mh_technique
    
    # Try to infer config from log if not provided
    if area is None or hw is None or mh_technique is None:
        # Look for file paths in log to infer parameters
        file_pattern = r'area-(\d+\.\d+)_hwscale-([\d.]+)_hwvar-[\d.]+_.*_assignment-(\w+)\.pkl'
        match = re.search(file_pattern, log_content)
        if match:
            if area is None:
                df['area'] = float(match.group(1))
            if hw is None:
                df['hw'] = float(match.group(2))
            if mh_technique is None:
                df['mh'] = match.group(3)
    
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_comparison_log.py <log_file> [area] [hw] [mh_technique]")
        print("\nExample:")
        print("  python parse_comparison_log.py slurm-12345.out 0.9 0.1 gl25")
        print("  python parse_comparison_log.py slurm-12345.out")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # Parse optional arguments
    area = float(sys.argv[2]) if len(sys.argv) > 2 else None
    hw = float(sys.argv[3]) if len(sys.argv) > 3 else None
    mh_technique = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Verify file exists
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"Parsing log file: {log_file}")
    print("-" * 60)
    
    # Parse the log
    df = parse_comparison_log(log_file, area, hw, mh_technique)
    
    print("-" * 60)
    print(f"Successfully extracted {len(df)} complete results")
    
    if len(df) > 0:
        # Generate output filename
        if 'area' in df.columns and 'hw' in df.columns and 'mh' in df.columns:
            area_val = df['area'].iloc[0]
            hw_val = df['hw'].iloc[0]
            mh_val = df['mh'].iloc[0]
            output_file = f'eval_logs/mh-vs-mip-area-{area_val:.2f}-hw-{hw_val:.2f}-{mh_val}-recovered.csv'
        else:
            output_file = f'eval_logs/mh-vs-mip-recovered.csv'
        
        # Save to CSV
        Path('eval_logs').mkdir(exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")
        
        # Display summary
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(df.to_string(index=False))
        
        if len(df) > 0:
            print("\n" + "=" * 60)
            print("AGGREGATE STATISTICS BY STRATEGY")
            print("=" * 60)
            summary = df.groupby('strategy').agg({
                'mip_cost': ['mean', 'std'],
                'mh_cost': ['mean', 'std'],
                'seed': 'count'
            }).round(2)
            summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
            print(summary)
    else:
        print("\nNo complete results found in log file.")


if __name__ == "__main__":
    main()
