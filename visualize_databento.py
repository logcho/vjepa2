#!/usr/bin/env python3
import os
import glob
import json
import zipfile
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import databento as db

# Set seaborn theme for premium look
sns.set_theme(style="darkgrid")

def find_zip_for_schema(schema, workspace_dir="."):
    """
    Dynamically locate the ZIP file containing the specified schema by inspecting metadata.json.
    """
    zip_files = glob.glob(os.path.join(workspace_dir, "EQUS-*.zip"))
    for zp in zip_files:
        try:
            with zipfile.ZipFile(zp, 'r') as zf:
                if 'metadata.json' in zf.namelist():
                    meta = json.loads(zf.read('metadata.json').decode('utf-8'))
                    if meta.get('query', {}).get('schema') == schema:
                        return zp
        except Exception:
            continue
    return None

def extract_and_load_dbn(schema, date, workspace_dir=".", extract_dir="databento_data"):
    """
    Finds the correct zip file for the schema, extracts the DBN file for the given date,
    and loads it into a pandas DataFrame.
    """
    os.makedirs(extract_dir, exist_ok=True)
    filename = f"equs-mini-{date.replace('-', '')}.{schema}.dbn.zst"
    dest_path = os.path.join(extract_dir, filename)
    
    if not os.path.exists(dest_path):
        print(f"File {filename} not found in {extract_dir}. Locating ZIP archive...")
        zip_path = find_zip_for_schema(schema, workspace_dir)
        if not zip_path:
            raise FileNotFoundError(f"Could not find ZIP archive containing schema '{schema}' in {workspace_dir}")
        
        print(f"Extracting {filename} from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if filename in zf.namelist():
                zf.extract(filename, extract_dir)
            else:
                available_dates = [f.split('-')[-1].split('.')[0] for f in zf.namelist() if f.endswith(f'.{schema}.dbn.zst')]
                raise FileNotFoundError(
                    f"Date {date} ({filename}) not found in zip archive. "
                    f"Available dates in this archive: {sorted(list(set(available_dates)))}"
                )
    
    print(f"Loading and parsing DBN file: {dest_path}...")
    store = db.DBNStore.from_file(dest_path)
    df = store.to_df()
    return df

def generate_visualization(date, symbol, rth_only=False, theme="dark", output_dir="databento_data"):
    """
    Generate premium visualization of Databento tick data.
    """
    # 1. Load Data (tbbo contains both trade details and top-of-book BBO bid/ask)
    try:
        df = extract_and_load_dbn("tbbo", date)
    except Exception as e:
        print(f"Error loading tbbo data: {e}")
        print("Falling back to trades schema...")
        try:
            df = extract_and_load_dbn("trades", date)
        except Exception as ex:
            print(f"Failed to load trades data: {ex}")
            return
    
    # Filter for the selected symbol
    df_sym = df[df['symbol'] == symbol].copy()
    if df_sym.empty:
        available_symbols = df['symbol'].unique().tolist()
        print(f"Error: Symbol {symbol} not found in dataset. Available symbols: {available_symbols}")
        return
    
    print(f"Processing {len(df_sym):,} records for {symbol} on {date}...")
    
    # 2. Timezone conversion (UTC -> US/Eastern)
    if df_sym.index.tz is None:
        df_sym.index = pd.to_datetime(df_sym.index, unit='ns').tz_localize('UTC')
    
    df_sym.index = df_sym.index.tz_convert('US/Eastern')
    
    # Regular Trading Hours (RTH) range definition
    rth_start = pd.Timestamp(f"{date} 09:30:00", tz='US/Eastern')
    rth_end = pd.Timestamp(f"{date} 16:00:00", tz='US/Eastern')
    
    # If rth_only is requested, filter df_sym
    if rth_only:
        df_sym = df_sym.between_time("09:30", "16:00")
        if df_sym.empty:
            print("Warning: No records found during Regular Trading Hours (09:30 - 16:00). Displaying full day instead.")
            df_sym = df[df['symbol'] == symbol].copy()
            df_sym.index = df_sym.index.tz_localize('UTC').tz_convert('US/Eastern')
            rth_only = False

    # Calculate additional metrics
    df_sym['spread'] = df_sym['ask_px_00'] - df_sym['bid_px_00']
    df_sym['mid_price'] = (df_sym['ask_px_00'] + df_sym['bid_px_00']) / 2.0
    
    # Cumulative trade volume
    df_sym['cum_volume'] = df_sym['size'].cumsum()
    
    # Configure theme colors
    if theme == "dark":
        plt.style.use('dark_background')
        bg_color = "#0f172a"  # Slate 900
        card_color = "#1e293b"  # Slate 800
        text_color = "#f8fafc"  # Slate 50
        grid_color = "#334155"  # Slate 700
        accent_color = "#818cf8"  # Indigo 400
        price_color = "#38bdf8"  # Sky 400
        bid_color = "#34d399"  # Emerald 400
        ask_color = "#f87171"  # Red 400
        spread_color = "#c084fc"  # Purple 400
        vol_color = "#fbbf24"  # Amber 400
    else:
        plt.style.use('default')
        bg_color = "#ffffff"
        card_color = "#f8fafc"
        text_color = "#0f172a"
        grid_color = "#e2e8f0"
        accent_color = "#4f46e5"
        price_color = "#0284c7"
        bid_color = "#059669"
        ask_color = "#dc2626"
        spread_color = "#7c3aed"
        vol_color = "#d97706"
        
    # Setup Figure and Subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, 
                             gridspec_kw={'height_ratios': [2.5, 1, 1.2]})
    fig.patch.set_facecolor(bg_color)
    
    for ax in axes:
        ax.set_facecolor(card_color)
        ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
        ax.tick_params(colors=text_color, labelsize=10)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
            
    # --- PANEL 1: Price and Bid-Ask Spread Band ---
    ax_price = axes[0]
    
    # Plot Bid-Ask Shading
    if 'bid_px_00' in df_sym.columns and 'ask_px_00' in df_sym.columns:
        ax_price.fill_between(df_sym.index, df_sym['bid_px_00'], df_sym['ask_px_00'], 
                              color=accent_color, alpha=0.25, label='BBO Bid-Ask Spread Band')
        ax_price.plot(df_sym.index, df_sym['bid_px_00'], color=bid_color, linestyle=':', alpha=0.7, label='Best Bid')
        ax_price.plot(df_sym.index, df_sym['ask_px_00'], color=ask_color, linestyle=':', alpha=0.7, label='Best Ask')
        
    # Plot Trade Prices
    ax_price.scatter(df_sym.index, df_sym['price'], color=price_color, s=8, alpha=0.8, label='Trade Price')
    
    # Highlight RTH with a shaded background if not filtering to RTH only
    if not rth_only and df_sym.index.min() < rth_start < df_sym.index.max():
        ax_price.axvspan(rth_start, min(rth_end, df_sym.index.max()), color=accent_color, alpha=0.05, label='Regular Trading Hours')
        ax_price.axvline(rth_start, color=accent_color, linestyle='--', alpha=0.4)
        if rth_end < df_sym.index.max():
            ax_price.axvline(rth_end, color=accent_color, linestyle='--', alpha=0.4)
            
    ax_price.set_ylabel("Price ($)", fontsize=12, fontweight='bold')
    ax_price.set_title(f"{symbol} Intraday Price & Order Book Top-of-Book ({date})", color=text_color, fontsize=14, fontweight='bold', pad=10)
    ax_price.legend(loc='upper left', framealpha=0.8, facecolor=card_color, edgecolor=grid_color, labelcolor=text_color)
    
    # --- PANEL 2: Bid-Ask Spread ---
    ax_spread = axes[1]
    if 'spread' in df_sym.columns and not df_sym['spread'].isnull().all():
        # Plot raw spread
        ax_spread.plot(df_sym.index, df_sym['spread'], color=spread_color, alpha=0.4, label='Spread')
        # Plot rolling mean (e.g. 50-tick window) to smooth
        rolling_spread = df_sym['spread'].rolling(window=min(50, len(df_sym))).mean()
        ax_spread.plot(df_sym.index, rolling_spread, color=spread_color, linewidth=2, label='50-Tick Rolling Avg')
        
        ax_spread.set_ylabel("Spread ($)", fontsize=12, fontweight='bold')
        ax_spread.legend(loc='upper left', framealpha=0.8, facecolor=card_color, edgecolor=grid_color, labelcolor=text_color)
    else:
        ax_spread.text(0.5, 0.5, "Spread data unavailable (trades-only file)", 
                      color=text_color, ha='center', va='center', transform=ax_spread.transAxes)
        ax_spread.set_ylabel("Spread ($)", fontsize=12, fontweight='bold')

    # --- PANEL 3: Volume and Cumulative Volume ---
    ax_vol = axes[2]
    # Trade size (Volume) as vertical stems/lines
    ax_vol.vlines(df_sym.index, 0, df_sym['size'], color=vol_color, alpha=0.6, label='Trade Size')
    ax_vol.set_ylabel("Trade Size (Shares)", fontsize=12, fontweight='bold')
    
    # Create cumulative volume on secondary y-axis
    ax_cum_vol = ax_vol.twinx()
    ax_cum_vol.plot(df_sym.index, df_sym['cum_volume'], color=accent_color, linewidth=2, label='Cumulative Volume')
    ax_cum_vol.set_ylabel("Cumulative Vol (Shares)", color=accent_color, fontsize=12, fontweight='bold')
    ax_cum_vol.tick_params(colors=accent_color)
    ax_cum_vol.grid(False)
    
    # Merge legends for ax_vol and ax_cum_vol
    lines1, labels1 = ax_vol.get_legend_handles_labels()
    lines2, labels2 = ax_cum_vol.get_legend_handles_labels()
    ax_vol.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.8, facecolor=card_color, edgecolor=grid_color, labelcolor=text_color)

    # Format X-axis Timestamps
    axes[2].set_xlabel("Time (US/Eastern)", fontsize=12, fontweight='bold')
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator, tz=df_sym.index.tz)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(formatter)
    
    # Rotate date labels
    plt.gcf().autofmt_xdate()
    
    # Statistics annotations
    avg_price = df_sym['price'].mean()
    total_vol = df_sym['size'].sum()
    avg_spread = df_sym['spread'].mean() if 'spread' in df_sym.columns else np.nan
    num_trades = len(df_sym)
    
    stat_text = (
        f"Trades: {num_trades:,}\n"
        f"Avg Price: ${avg_price:.2f}\n"
        f"Total Vol: {total_vol:,} shs"
    )
    if not np.isnan(avg_spread):
        stat_text += f"\nAvg Spread: ${avg_spread*100:.2f}¢"
        
    props = dict(boxstyle='round', facecolor=card_color, alpha=0.9, edgecolor=grid_color)
    axes[0].text(0.98, 0.95, stat_text, transform=axes[0].transAxes, fontsize=10,
                color=text_color, verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Layout Adjustments
    plt.tight_layout()
    
    # Save Image
    os.makedirs(output_dir, exist_ok=True)
    rth_suffix = "_rth" if rth_only else ""
    output_filename = f"{symbol}_{date.replace('-', '')}{rth_suffix}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization successfully saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and visualize Databento market data.")
    parser.add_argument("--date", type=str, default="2025-12-22", help="Date in YYYY-MM-DD format (default: 2025-12-22)")
    parser.add_argument("--symbol", type=str, default="TSLA", help="Ticker symbol to visualize (default: TSLA)")
    parser.add_argument("--rth-only", action="store_true", help="Filter for Regular Trading Hours (09:30 - 16:00 EST)")
    parser.add_argument("--theme", type=str, choices=["light", "dark"], default="dark", help="Visualization theme (default: dark)")
    
    args = parser.parse_args()
    generate_visualization(args.date, args.symbol, args.rth_only, args.theme)
