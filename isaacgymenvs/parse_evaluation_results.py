#!/usr/bin/env python3
"""
Parse evaluation logs and generate comprehensive statistics.
Usage: python parse_evaluation_results.py <log_directory>
"""

import os
import sys
import re
from pathlib import Path

def parse_log_file(log_path):
    """Extract metrics from a single log file."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Extract success rate
    success_match = re.search(r'final success_rate:\s*([\d.]+)\s*\(([\d.]+)%\)', content)
    if success_match:
        results['success_rate'] = float(success_match.group(1))
        results['success_rate_pct'] = float(success_match.group(2))
    
    # Extract mean placement distance
    dist_match = re.search(r'final mean_placement_dist:\s*([\d.]+)cm\s*\(([\d.]+)mm\)', content)
    if dist_match:
        results['mean_placement_dist_cm'] = float(dist_match.group(1))
        results['mean_placement_dist_mm'] = float(dist_match.group(2))
    
    # Extract av reward
    reward_match = re.search(r'av reward:\s*([\d.]+)', content)
    if reward_match:
        results['avg_reward'] = float(reward_match.group(1))
    
    # Extract av steps
    steps_match = re.search(r'av steps:\s*([\d.]+)', content)
    if steps_match:
        results['avg_steps'] = float(steps_match.group(1))
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_evaluation_results.py <log_directory>")
        sys.exit(1)
    
    log_dir = Path(sys.argv[1])
    
    if not log_dir.exists():
        print(f"Error: Directory {log_dir} does not exist")
        sys.exit(1)
    
    # Find all seed log files
    log_files = sorted(log_dir.glob("seed_*.log"))
    
    if not log_files:
        print(f"No seed_*.log files found in {log_dir}")
        sys.exit(1)
    
    print(f"Found {len(log_files)} seed log files\n")
    print("="*80)
    
    all_results = []
    
    # Parse each log file
    for log_file in log_files:
        seed = log_file.stem.replace("seed_", "")
        results = parse_log_file(log_file)
        
        if results:
            results['seed'] = seed
            all_results.append(results)
    
    # Print table for Excel
    if all_results:
        print("\n" + "="*80)
        print("EXCEL-READY TABLE (Copy the table below):")
        print("="*80 + "\n")
        
        # Print header
        print("seed\tsuccess_rate\tl2_dist\tavg_reward\tavg_steps")
        
        # Print each seed's data
        success_rates = []
        l2_dists = []
        avg_rewards = []
        avg_steps = []
        
        for result in all_results:
            seed = result.get('seed', 'N/A')
            success_rate = result.get('success_rate_pct', None)
            l2_dist = result.get('mean_placement_dist_mm', None)
            avg_reward = result.get('avg_reward', None)
            avg_step = result.get('avg_steps', None)
            
            if success_rate is not None:
                success_rates.append(success_rate)
            if l2_dist is not None:
                l2_dists.append(l2_dist)
            if avg_reward is not None:
                avg_rewards.append(avg_reward)
            if avg_step is not None:
                avg_steps.append(avg_step)
            
            success_str = f"{success_rate:.2f}" if success_rate is not None else "N/A"
            l2_str = f"{l2_dist:.2f}" if l2_dist is not None else "N/A"
            reward_str = f"{avg_reward:.2f}" if avg_reward is not None else "N/A"
            step_str = f"{avg_step:.2f}" if avg_step is not None else "N/A"
            
            print(f"{seed}\t{success_str}\t{l2_str}\t{reward_str}\t{step_str}")
        
        # Calculate statistics
        if success_rates and l2_dists:
            # Mean row
            mean_success = sum(success_rates) / len(success_rates)
            mean_l2 = sum(l2_dists) / len(l2_dists)
            mean_reward = sum(avg_rewards) / len(avg_rewards) if avg_rewards else 0
            mean_step = sum(avg_steps) / len(avg_steps) if avg_steps else 0
            print(f"Mean\t{mean_success:.2f}\t{mean_l2:.2f}\t{mean_reward:.2f}\t{mean_step:.2f}")
            
            # Std row
            std_success = (sum((x - mean_success)**2 for x in success_rates) / len(success_rates))**0.5
            std_l2 = (sum((x - mean_l2)**2 for x in l2_dists) / len(l2_dists))**0.5
            std_reward = (sum((x - mean_reward)**2 for x in avg_rewards) / len(avg_rewards))**0.5 if avg_rewards else 0
            std_step = (sum((x - mean_step)**2 for x in avg_steps) / len(avg_steps))**0.5 if avg_steps else 0
            print(f"Std\t{std_success:.2f}\t{std_l2:.2f}\t{std_reward:.2f}\t{std_step:.2f}")
        
        print("\n" + "="*80)
        print("Copy the table above and paste directly into Excel")
        print("="*80)
        
        # Save detailed results to CSV
        csv_path = log_dir / "aggregate_results.csv"
        with open(csv_path, 'w') as f:
            # Header
            headers = ['seed', 'success_rate_pct', 'mean_placement_dist_mm', 'avg_reward', 'avg_steps']
            f.write(','.join(headers) + '\n')
            
            # Data rows
            for result in all_results:
                row = [
                    result.get('seed', ''),
                    str(result.get('success_rate_pct', '')),
                    str(result.get('mean_placement_dist_mm', '')),
                    str(result.get('avg_reward', '')),
                    str(result.get('avg_steps', ''))
                ]
                f.write(','.join(row) + '\n')
        
        print(f"\n{'='*80}")
        print(f"Detailed results saved to: {csv_path}")
        print("="*80)

if __name__ == "__main__":
    main()
