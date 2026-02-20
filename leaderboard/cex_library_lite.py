# cex_library library

import pandas as pd
import requests
import time
import os
import logging
import numpy as np
from datetime import datetime

class KrakenCEX:
    def __init__(self, storage_dir="market_data/kraken"):
        self.base_url = "https://api.kraken.com/0/public/Trades"
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def _get_filename(self, pair, timestamp):
        # Creates a reusable daily file path: market_data/kraken/XETHZUSD_2026-02-04.parquet
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        return os.path.join(self.storage_dir, f"{pair}_{date_str}.parquet")

    def fetch_tick_history(self, pair="XETHZUSD", start_ts=None, end_ts=None):
        current_start = int(start_ts)
        final_df = pd.DataFrame()

        while current_start < end_ts:
            file_path = self._get_filename(pair, current_start)
            
            if os.path.exists(file_path):
                try:
                    logging.info(f"📂 Loading local partition: {file_path}")
                    day_df = pd.read_parquet(file_path)
                except Exception as e:
                    logging.warning(f"⚠️ Corrupted parquet {file_path}: {e} — deleting and re-fetching.")
                    os.remove(file_path)
                    day_df = self._crawl_window(pair, current_start, current_start + 86400)
                    if not day_df.empty:
                        day_df.to_parquet(file_path, compression='snappy')
            else:
                logging.info(f"🌐 Fetching missing day from Kraken API...")
                day_df = self._crawl_window(pair, current_start, current_start + 86400)
                if not day_df.empty:
                    day_df.to_parquet(file_path, compression='snappy')
            
            if not day_df.empty:
                final_df = pd.concat([final_df, day_df], ignore_index=True)
            current_start += 86400

        if final_df.empty:
            return final_df

        # Re-parse dt_utc — stored as string in parquet to avoid ns/us precision conflicts
        final_df['dt_utc'] = pd.to_datetime(final_df['dt_utc'], utc=True)

        # Trim to requested window and deduplicate
        mask = (final_df['ts'] >= start_ts) & (final_df['ts'] <= end_ts)
        return final_df[mask].drop_duplicates(subset=['id']).sort_values('ts')

    def _crawl_window(self, pair, start, end):
        all_ticks = []
        since = f"{int(start)}000000000" # Kraken nanoseconds
        while True:
            res = requests.get(self.base_url, params={'pair': pair, 'since': since}).json()
            if 'result' not in res or not res['result'][pair]: break
            all_ticks.extend(res['result'][pair])
            new_since = res['result']['last']
            if int(new_since) >= (end * 10**9) or new_since == since: break
            since = new_since
            time.sleep(1.1)
        
        df = pd.DataFrame(all_ticks, columns=['price', 'vol', 'ts', 'side', 'type', 'misc', 'id'])
        df['price'] = df['price'].astype(float)
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        # Kraken returns ts in seconds (float). Nanosecond values > 1e12 indicate
        # a malformed row — drop them rather than letting them overflow datetime range.
        df = df[df['ts'].notna() & (df['ts'] < 1e12)].copy()
        df['dt_utc'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        return df

    def get_market_signals(self, tick_df, bps_threshold=5):
        # Resample to 1s slots to align with block-based DEX execution
        # Cap forward-fill at 60s — stale prices beyond this are unreliable for alpha detection
        signals = tick_df.set_index('dt_utc').resample('1s').last().ffill(limit=60)
        signals['log_ret'] = np.log(signals['price'] / signals['price'].shift(1))
        signals['bps'] = (signals['log_ret'] * 10000).fillna(0)
        return signals[signals['bps'].abs() >= bps_threshold]

def align_ledger_to_cex(ledger_df, cex_df):
    # Ensure the DEX ledger is sorted by time
    ledger_df = ledger_df.sort_values('dt_utc')
    
    # FIX: Use sort_index() because cex_df is indexed by 'dt_utc'
    # merge_asof REQUIREMENT: Both dataframes must be sorted by the key
    cex_df = cex_df.sort_index() 
    
    return pd.merge_asof(
        ledger_df, 
        cex_df[['price', 'bps']], 
        left_on='dt_utc', 
        right_index=True, 
        direction='backward'
    )
