#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Set seaborn theme for premium look
sns.set_theme(style="darkgrid")

def generate_timeline_visualization(input_csv="daily_aggregates.csv", theme="dark", output_dir="databento_data"):
    """
    Loads daily_aggregates.csv and generates a 3-panel timeline visualization
    spanning the entire 6-month historical dataset.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found. Please run compute_daily_aggregates.py first.")
        return
        
    print(f"Loading daily aggregates from {input_csv}...")
    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract unique symbols
    symbols = sorted(df['symbol'].unique().tolist())
    print(f"Aggregated symbols found: {symbols}")
    
    # Configure theme colors
    if theme == "dark":
        plt.style.use('dark_background')
        bg_color = "#0f172a"  # Slate 900
        card_color = "#1e293b"  # Slate 800
        text_color = "#f8fafc"  # Slate 50
        grid_color = "#334155"  # Slate 700
    else:
        plt.style.use('default')
        bg_color = "#ffffff"
        card_color = "#f8fafc"
        text_color = "#0f172a"
        grid_color = "#e2e8f0"
        
    # Palettes for the 5 symbols
    palette = sns.color_palette("husl", len(symbols))
    symbol_colors = {symbols[i]: palette[i] for i in range(len(symbols))}
    
    # Setup Figure and Subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.patch.set_facecolor(bg_color)
    
    for ax in axes:
        ax.set_facecolor(card_color)
        ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
        ax.tick_params(colors=text_color, labelsize=10)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
            
    # --- PANEL 1: Cumulative Performance (%) ---
    ax_perf = axes[0]
    for sym in symbols:
        df_sym = df[df['symbol'] == sym].sort_values('date').copy()
        if df_sym.empty:
            continue
        
        # Calculate cumulative returns starting from the first close price
        first_close = df_sym['close'].iloc[0]
        df_sym['cum_return'] = (df_sym['close'] / first_close - 1.0) * 100.0
        
        ax_perf.plot(df_sym['date'], df_sym['cum_return'], 
                     label=f"{sym} (Final: {df_sym['cum_return'].iloc[-1]:+.1f}%)", 
                     color=symbol_colors[sym], linewidth=2.5)
        
    ax_perf.axhline(0, color=text_color, linestyle='--', alpha=0.3)
    ax_perf.set_ylabel("Cumulative Return (%)", fontsize=12, fontweight='bold')
    ax_perf.set_title("Multi-Asset Price Performance Timeline (Dec 2025 - Jun 2026)", 
                      color=text_color, fontsize=15, fontweight='bold', pad=12)
    ax_perf.legend(loc='upper left', framealpha=0.9, facecolor=card_color, edgecolor=grid_color, labelcolor=text_color, fontsize=11)
    
    # --- PANEL 2: Daily Trading Volume (Rolling Avg) ---
    ax_vol = axes[1]
    for sym in symbols:
        df_sym = df[df['symbol'] == sym].sort_values('date').copy()
        if df_sym.empty:
            continue
        
        # 5-day rolling average to smooth daily spikes
        df_sym['volume_ma'] = df_sym['volume'].rolling(window=5, min_periods=1).mean()
        
        ax_vol.plot(df_sym['date'], df_sym['volume_ma'] / 1e6, 
                    label=sym, color=symbol_colors[sym], linewidth=2, alpha=0.8)
        
    ax_vol.set_ylabel("Daily Volume (M shs, 5D MA)", fontsize=11, fontweight='bold')
    ax_vol.legend(loc='upper left', framealpha=0.9, facecolor=card_color, edgecolor=grid_color, labelcolor=text_color, fontsize=10)
    
    # --- PANEL 3: Daily Average Bid-Ask Spread (¢) ---
    ax_spread = axes[2]
    for sym in symbols:
        df_sym = df[df['symbol'] == sym].sort_values('date').copy()
        if df_sym.empty or df_sym['avg_spread'].isnull().all():
            continue
            
        # Convert dollar spread to cents and smooth with 5-day rolling mean
        df_sym['spread_cents_ma'] = (df_sym['avg_spread'] * 100).rolling(window=5, min_periods=1).mean()
        
        ax_spread.plot(df_sym['date'], df_sym['spread_cents_ma'], 
                       label=sym, color=symbol_colors[sym], linewidth=2, alpha=0.8)
        
    ax_spread.set_ylabel("Avg Bid-Ask Spread (¢, 5D MA)", fontsize=11, fontweight='bold')
    ax_spread.legend(loc='upper left', framealpha=0.9, facecolor=card_color, edgecolor=grid_color, labelcolor=text_color, fontsize=10)
    
    # Format X-axis Dates
    axes[2].set_xlabel("Date", fontsize=12, fontweight='bold')
    ax_spread.xaxis.set_major_locator(mdates.MonthLocator())
    ax_spread.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Save Figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "timeline_overview.png")
    plt.savefig(output_path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    plt.close()
    
    print(f"Timeline overview successfully saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Databento daily aggregates timeline.")
    parser.add_argument("--input", type=str, default="daily_aggregates.csv", help="Input CSV path (default: daily_aggregates.csv)")
    parser.add_argument("--theme", type=str, choices=["light", "dark"], default="dark", help="Visualization theme (default: dark)")
    
    args = parser.parse_args()
    generate_timeline_visualization(args.input, args.theme)
