#!/usr/bin/env python3
import os
import glob
import json
import zipfile
import shutil
import pandas as pd
import numpy as np
import databento as db

def find_zip_for_schema(schema, workspace_dir="."):
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

def main():
    schema = "tbbo"
    workspace_dir = "."
    temp_dir = "databento_temp"
    output_csv = "daily_aggregates.csv"
    
    zip_path = find_zip_for_schema(schema, workspace_dir)
    if not zip_path:
        print(f"Error: Could not find ZIP archive for schema '{schema}'")
        return
        
    print(f"Found ZIP archive: {zip_path}")
    
    os.makedirs(temp_dir, exist_ok=True)
    
    daily_records = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of all daily DBN files in the zip
        dbn_files = sorted([f for f in zf.namelist() if f.endswith(f".{schema}.dbn.zst")])
        total_files = len(dbn_files)
        print(f"Discovered {total_files} daily DBN files. Starting aggregation...")
        
        for idx, filename in enumerate(dbn_files):
            # Extract date from filename (e.g. equs-mini-20251222.tbbo.dbn.zst -> 2025-12-22)
            # format: equs-mini-YYYYMMDD.tbbo.dbn.zst
            date_str = filename.split('-')[-1].split('.')[0]
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            
            print(f"[{idx+1}/{total_files}] Processing {date_formatted}...", end="", flush=True)
            
            # Extract file to temp folder
            temp_file_path = os.path.join(temp_dir, filename)
            zf.extract(filename, temp_dir)
            
            try:
                # Load DBN and convert to DataFrame
                store = db.DBNStore.from_file(temp_file_path)
                df = store.to_df()
                
                # Check for required columns
                has_spread = 'bid_px_00' in df.columns and 'ask_px_00' in df.columns
                if has_spread:
                    df['spread'] = df['ask_px_00'] - df['bid_px_00']
                
                # Process each symbol in the file
                unique_symbols = df['symbol'].unique()
                for sym in unique_symbols:
                    df_sym = df[df['symbol'] == sym]
                    if df_sym.empty:
                        continue
                        
                    # Calculate OHLC from trade prices
                    prices = df_sym['price'].values
                    open_px = float(prices[0])
                    close_px = float(prices[-1])
                    high_px = float(prices.max())
                    low_px = float(prices.min())
                    
                    # Calculate Volume and Tick count
                    volume = int(df_sym['size'].sum())
                    ticks = len(df_sym)
                    
                    # Calculate average spread
                    avg_spread = float(df_sym['spread'].mean()) if has_spread else np.nan
                    
                    daily_records.append({
                        'date': date_formatted,
                        'symbol': sym,
                        'open': open_px,
                        'high': high_px,
                        'low': low_px,
                        'close': close_px,
                        'volume': volume,
                        'avg_spread': avg_spread,
                        'ticks': ticks
                    })
                print(" Success.")
            except Exception as e:
                print(f" Failed with error: {e}")
            finally:
                # Clean up extracted temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Save to CSV
    if daily_records:
        df_aggs = pd.DataFrame(daily_records)
        df_aggs.sort_values(by=['date', 'symbol'], inplace=True)
        df_aggs.to_csv(output_csv, index=False)
        print(f"\nSaved daily aggregates to: {output_csv}")
        print(f"Total rows: {len(df_aggs)}")
    else:
        print("\nNo records were aggregated.")

if __name__ == "__main__":
    main()
