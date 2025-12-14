#!/usr/bin/env python3
"""
Data Export Utility

CLI utility for exporting research data, metrics,
and system information for analysis.
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import settings
from utils.logger import StructuredLogger


def export_metrics(output_dir: str = "research_data"):
    """Export healing metrics to CSV and JSON."""
    print("Exporting healing metrics...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_file = settings.metrics_file
    if os.path.exists(metrics_file):
        # Export JSON
        json_output = os.path.join(output_dir, "healing_metrics.json")
        with open(metrics_file, 'r') as src, open(json_output, 'w') as dst:
            dst.write(src.read())
        print(f"✓ Exported metrics to {json_output}")
        
        # Export CSV
        csv_output = os.path.join(output_dir, "healing_metrics.csv")
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            with open(csv_output, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Metric', 'Value'])
                
                for key, value in metrics.items():
                    if key != 'detailed_logs':
                        writer.writerow([key, value])
            
            print(f"✓ Exported metrics CSV to {csv_output}")
        except Exception as e:
            print(f"✗ Error exporting CSV: {e}")
    else:
        print("✗ No metrics file found")


def export_logs(output_dir: str = "research_data"):
    """Export system logs to CSV."""
    print("Exporting system logs...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = settings.log_file
    if os.path.exists(log_file):
        csv_output = os.path.join(output_dir, "system_logs.csv")
        
        try:
            with open(csv_output, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'level', 'logger', 'message', 'agent', 'error_type'])
                
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            writer.writerow([
                                log_entry.get('timestamp', ''),
                                log_entry.get('level', ''),
                                log_entry.get('logger', ''),
                                log_entry.get('message', ''),
                                log_entry.get('agent', ''),
                                log_entry.get('error_type', '')
                            ])
                        except:
                            continue
            
            print(f"✓ Exported logs to {csv_output}")
        except Exception as e:
            print(f"✗ Error exporting logs: {e}")
    else:
        print("✗ No log file found")


def export_collected_data(output_dir: str = "research_data"):
    """Export collected data files."""
    print("Exporting collected data...")
    
    data_dir = "data/collected_data"
    output_data_dir = os.path.join(output_dir, "collected_data")
    
    if os.path.exists(data_dir):
        import shutil
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)
        shutil.copytree(data_dir, output_data_dir)
        print(f"✓ Exported collected data to {output_data_dir}")
    else:
        print("✗ No collected data directory found")


def generate_report(output_dir: str = "research_data"):
    """Generate a summary report."""
    print("Generating summary report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, "summary_report.md")
    
    try:
        with open(report_file, 'w') as f:
            f.write("# ETL Sentiment System Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # System info
            f.write("## System Information\n\n")
            f.write(f"- Version: {settings.app_version}\n")
            f.write(f"- Ollama Model: {settings.ollama_model}\n")
            f.write(f"- Healing Enabled: {settings.healing_enabled}\n\n")
            
            # Metrics summary
            metrics_file = settings.metrics_file
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as mf:
                    metrics = json.load(mf)
                
                f.write("## Healing Metrics\n\n")
                successful = metrics.get('successful_repairs', 0)
                failed = metrics.get('failed_repairs', 0)
                total = successful + failed
                
                if total > 0:
                    success_rate = (successful / total) * 100
                    avg_mttr = metrics.get('total_mttr', 0) / successful if successful > 0 else 0
                    
                    f.write(f"- Total Repairs: {total}\n")
                    f.write(f"- Successful: {successful}\n")
                    f.write(f"- Failed: {failed}\n")
                    f.write(f"- Success Rate: {success_rate:.1f}%\n")
                    f.write(f"- Average MTTR: {avg_mttr:.2f} seconds\n")
                else:
                    f.write("No healing operations recorded yet\n")
                f.write("\n")
        
        print(f"✓ Generated report at {report_file}")
    except Exception as e:
        print(f"✗ Error generating report: {e}")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETL Sentiment Data Export")
    parser.add_argument("--output", "-o", default="research_data", 
                       help="Output directory (default: research_data)")
    parser.add_argument("--metrics", action="store_true", help="Export metrics only")
    parser.add_argument("--logs", action="store_true", help="Export logs only")
    parser.add_argument("--data", action="store_true", help="Export collected data only")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    
    args = parser.parse_args()
    
    print(f"Exporting data to: {args.output}")
    print()
    
    if args.metrics:
        export_metrics(args.output)
    elif args.logs:
        export_logs(args.output)
    elif args.data:
        export_collected_data(args.output)
    elif args.report:
        generate_report(args.output)
    else:
        # Export everything
        export_metrics(args.output)
        export_logs(args.output)
        export_collected_data(args.output)
        generate_report(args.output)
    
    print()
    print("Export complete!")


if __name__ == "__main__":
    main()