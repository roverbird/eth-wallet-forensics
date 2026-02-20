from __future__ import annotations

"""
leaderboard.py — MEV / DEX wallet leaderboard generator
========================================================

Architecture (1 GB VPS safe)
-----------------------------
- Two-pass streaming design: no full DataFrame is ever held in RAM.
  Pass 1 deduplicates and sorts row keys into a lightweight temp parquet index.
  Pass 2 replays rows in global order, one buffered chunk at a time.
- CEX tick history is resampled to 1-second signals before any further use,
  collapsing ~700k raw ticks/week to ~60k signal rows (~5 MB).
- Peak RAM budget: 1 chunk (~10 MB) + cex_signals (~5 MB) + wallets dict.

Performance metrics
--------------------
win_rate          : fraction of blocks where equity increased vs previous block.
                    Robust to position size; 0.5 = random, 1.0 = never lost a block.
profit_per_trade  : net_usdc_after_gas / n_trades. Capital efficiency metric —
                    how much value is extracted per swap regardless of total volume.
profit_factor     : gross_block_profit / gross_block_loss across all MTM intervals.
                    > 1.0 = profitable system. Common in trading system evaluation;
                    robust to non-normality, no benchmark rate assumption needed.

CEX-relative columns
---------------------
mean_cex_bps_at_trade : average CEX momentum (bps) at the moment of each swap.
                        Positive on BUY_ETH = wallet bought into rising CEX price.
                        Strong signal for arbitrageurs vs noise traders.
informed_trade_pct    : fraction of trades where direction matched CEX momentum
                        (BUY_ETH with bps > 0, or SELL_ETH with bps < 0).
                        ~70%+ indicates CEX-informed / arb flow.
cex_slippage_bps      : mean (dex_price − cex_price) / cex_price × 10000.
                        Negative = wallet paid more than CEX (noise trader).
                        Near-zero or positive = tight or price-improved execution.

Filtering
----------
Wallets with fewer than MIN_TRADES trades are written to a separate
low_activity CSV rather than the main leaderboard, so they are not lost
but do not pollute ranked results with statistically meaningless metrics.
"""

import gc
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from cex_library_lite import KrakenCEX, align_ledger_to_cex

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE        = "forensic_dex_data_v2.csv"
OUTPUT_FILE_DAILY   = "leaderboard_static_daily.csv"
OUTPUT_FILE_WEEKLY  = "leaderboard_static_weekly.csv"
LOW_ACTIVITY_SUFFIX = "_low_activity"          # inserted before .csv on each window file
CHUNK_SIZE          = 50_000   # rows per CSV read — ~10 MB at ~200 B/row
CEX_PAIR            = "XETHZUSD"
BPS_THRESHOLD       = 5        # minimum CEX move (bps) to count as a signal
BUFFER_BLOCKS       = 500      # blocks to buffer in Pass 2 before flushing
MIN_TRADES          = 20       # wallets below this go to the low-activity file

# Rolling windows — add rows here to introduce new periods without touching pipeline code
# (days, output_file, human-readable label)
WINDOWS: List[Tuple[int, str, str]] = [
    (1, OUTPUT_FILE_DAILY,  "daily"),
    (7, OUTPUT_FILE_WEEKLY, "weekly"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

TARGET_COLS = [
    "block", "log_index", "timestamp", "swap_sender", "side",
    "base_qty_eth", "quote_qty_usdc", "execution_price",
    "tx_fee_eth", "tx_hash",
]


# ---------------------------------------------------------------------------
# WalletStats
# ---------------------------------------------------------------------------

class WalletStats:
    """
    Per-wallet inventory, P&L, and execution metrics.

    Assumptions
    -----------
    - Zero starting position for all wallets.
    - mark_to_market() called once per block (not per trade).
    - avg_log_index: mean Ethereum receipt log-index. Persistently low values
      (0–2) suggest the wallet is the first event in a transaction bundle,
      a common front-run / sandwich indicator.
    - Win rate, profit factor, and profit_per_trade replace Sharpe ratio.
      Sharpe is inappropriate here: MEV wallets extract value atomically
      rather than holding market exposure, so return variance is not a
      meaningful risk proxy.
    - All Welford / running accumulators avoid storing per-trade lists.
    """

    __slots__ = (
        "eth_position", "usdc_balance", "net_usdc_extracted", "gas_spent_usdc",
        "_initial_equity", "_last_equity", "_peak_equity", "max_drawdown",
        # win rate & profit factor accumulators
        "_n_blocks", "_n_winning_blocks",
        "_gross_profit", "_gross_loss",
        "_log_index_sum", "_log_index_count", "last_price",
        # CEX signal accumulators
        "_bps_n", "_bps_mean",
        "_informed_count",
        "_slippage_n", "_slippage_mean",
    )

    def __init__(self):
        self.eth_position          = 0.0
        self.usdc_balance          = 0.0
        self.net_usdc_extracted    = 0.0
        self.gas_spent_usdc        = 0.0

        self._initial_equity: Optional[float] = None
        self._last_equity: Optional[float]    = None
        self._peak_equity                     = 0.0
        self.max_drawdown                     = 0.0

        # Win rate & profit factor
        self._n_blocks         = 0
        self._n_winning_blocks = 0
        self._gross_profit     = 0.0   # sum of positive block-level equity changes
        self._gross_loss       = 0.0   # sum of absolute negative block-level changes

        self._log_index_sum   = 0.0
        self._log_index_count = 0
        self.last_price       = 0.0

        self._bps_n          = 0
        self._bps_mean       = 0.0
        self._informed_count = 0
        self._slippage_n     = 0
        self._slippage_mean  = 0.0

    # ------------------------------------------------------------------
    # Trade ingestion
    # ------------------------------------------------------------------

    def process_trade(
        self,
        side: str,
        eth_qty: float,
        usdc_qty: float,
        log_index: int,
        gas_fee_eth: float,
        price: float,
        cex_bps: Optional[float] = None,
        cex_price: Optional[float] = None,
    ) -> None:
        """
        Update inventory for one swap. Call for every trade in a block,
        then call mark_to_market() once at end of block.

        cex_bps   — CEX log-return in bps at trade time (None if unavailable).
        cex_price — CEX mid price at trade time for slippage calculation.
        """
        if side == "BUY_ETH":
            self.eth_position       += eth_qty
            self.usdc_balance       -= usdc_qty
            self.net_usdc_extracted -= usdc_qty
        else:  # SELL_ETH
            self.eth_position       -= eth_qty
            self.usdc_balance       += usdc_qty
            self.net_usdc_extracted += usdc_qty

        self.gas_spent_usdc += gas_fee_eth * price
        self.last_price      = price

        self._log_index_sum   += log_index
        self._log_index_count += 1

        # CEX momentum: Welford update
        if cex_bps is not None and not np.isnan(cex_bps):
            self._bps_n    += 1
            self._bps_mean += (cex_bps - self._bps_mean) / self._bps_n
            if (side == "BUY_ETH" and cex_bps > 0) or \
               (side == "SELL_ETH" and cex_bps < 0):
                self._informed_count += 1

        # CEX slippage: Welford update
        if cex_price is not None and not np.isnan(cex_price) and cex_price > 0:
            slip_bps = ((price - cex_price) / cex_price) * 10_000
            self._slippage_n    += 1
            self._slippage_mean += (slip_bps - self._slippage_mean) / self._slippage_n

    # ------------------------------------------------------------------
    # Per-block mark-to-market — call ONCE per block
    # ------------------------------------------------------------------

    def mark_to_market(self, price: float) -> None:
        equity      = self.usdc_balance + self.eth_position * price
        self.last_price = price

        if self._initial_equity is None:
            self._initial_equity = equity
            self._last_equity    = equity
            self._peak_equity    = equity
            return

        prev   = self._last_equity
        change = equity - prev

        if abs(prev) > 0.01:
            self._n_blocks += 1
            if change > 0:
                self._n_winning_blocks += 1
                self._gross_profit     += change
            elif change < 0:
                self._gross_loss       += abs(change)

        if equity > self._peak_equity:
            self._peak_equity = equity
        dd = self._peak_equity - equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        self._last_equity = equity

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def finalize(self) -> dict:
        last_equity    = self._last_equity    or 0.0
        initial_equity = self._initial_equity or 0.0
        net_pnl        = last_equity - initial_equity
        net_after_gas  = self.net_usdc_extracted - self.gas_spent_usdc

        win_rate = (
            round(self._n_winning_blocks / self._n_blocks, 4)
            if self._n_blocks else None
        )
        # profit_factor: gross_profit / gross_loss across block-level MTM changes.
        # inf means wallet never had a losing block. None means no activity.
        profit_factor = (
            round(self._gross_profit / self._gross_loss, 4)
            if self._gross_loss > 0
            else (None if self._gross_profit == 0 else float("inf"))
        )
        profit_per_trade = (
            round(net_after_gas / self._log_index_count, 4)
            if self._log_index_count else None
        )

        return {
            "net_pnl":               round(net_pnl, 4),
            "current_equity":        round(last_equity, 4),
            "net_usdc_extracted":    round(self.net_usdc_extracted, 4),
            "net_usdc_after_gas":    round(net_after_gas, 4),
            "gas_spent_usdc":        round(self.gas_spent_usdc, 4),
            "win_rate":              win_rate,
            "profit_factor":         profit_factor,
            "profit_per_trade":      profit_per_trade,
            "max_drawdown":          round(self.max_drawdown, 4),
            "avg_log_index":         round(self._log_index_sum / self._log_index_count, 2)
                                     if self._log_index_count else 0.0,
            "eth_position":          round(self.eth_position, 8),
            "usdc_balance":          round(self.usdc_balance, 4),
            "n_trades":              self._log_index_count,
            "n_blocks":              self._n_blocks,
            # CEX-relative metrics (None when no CEX data available)
            "mean_cex_bps_at_trade": round(self._bps_mean, 2)      if self._bps_n      else None,
            "informed_trade_pct":    round(self._informed_count / self._bps_n, 4)
                                     if self._bps_n else None,
            "cex_slippage_bps":      round(self._slippage_mean, 2) if self._slippage_n else None,
        }


# ---------------------------------------------------------------------------
# CEX signal preparation
# ---------------------------------------------------------------------------

def _build_cex_signals(
    kraken: KrakenCEX,
    cutoff: datetime,
) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Fetch CEX ticks, resample to 1-second signal slots, return
    (signals_df, final_price). Raw ticks are freed immediately after
    resampling to keep RAM usage low.

    signals_df is indexed by dt_utc with columns: price, bps.
    """
    now_ts = datetime.now(timezone.utc).timestamp()
    try:
        cex_ticks = kraken.fetch_tick_history(CEX_PAIR, cutoff.timestamp(), now_ts)
    except Exception as exc:
        log.error("CEX fetch failed: %s — CEX columns will be null.", exc)
        return None, None

    if cex_ticks.empty:
        log.warning("CEX tick history empty — CEX columns will be null.")
        return None, None

    final_price = float(cex_ticks.iloc[-1]["price"])
    log.info("CEX final mark: %.2f  |  %d raw ticks", final_price, len(cex_ticks))

    signals = kraken.get_market_signals(cex_ticks, bps_threshold=BPS_THRESHOLD)
    del cex_ticks
    gc.collect()

    log.info(
        "CEX signals post-resample+filter: %d rows  (%.1f MB)",
        len(signals),
        signals.memory_usage(deep=True).sum() / 1e6,
    )
    return signals, final_price


# ---------------------------------------------------------------------------
# Pass 1 — lightweight sorted/deduplicated index
# ---------------------------------------------------------------------------

def _build_sorted_index(cutoff: datetime) -> Optional[str]:
    """
    Scan INPUT_FILE in chunks, keep only key columns for active rows,
    deduplicate on (tx_hash, log_index), sort by (block, log_index),
    and write to a temp parquet. Returns the temp file path.

    Memory: index columns only (~40 bytes/row vs ~200 bytes full row).
    """
    index_cols = ["block", "log_index", "timestamp", "tx_hash"]
    frames = []

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, usecols=index_cols):
        chunk["dt_utc"] = pd.to_datetime(chunk["timestamp"], utc=True)
        active = chunk.loc[
            chunk["dt_utc"] > cutoff,
            ["block", "log_index", "tx_hash", "dt_utc"]
        ]
        if not active.empty:
            frames.append(active)

    if not frames:
        log.warning("No rows found within the requested window (cutoff: %s).", cutoff.isoformat())
        return None

    idx = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset=["tx_hash", "log_index"])
          .sort_values(["block", "log_index"])
          .reset_index(drop=True)
    )
    del frames
    gc.collect()

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    idx.to_parquet(tmp.name, index=False)
    log.info("Pass 1: %d unique trades indexed → %s", len(idx), tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Pass 2 — stream full rows in globally-sorted order
# ---------------------------------------------------------------------------

def _stream_sorted_chunks(index_path: str):
    """
    Generator. Yields DataFrames of full rows in globally-sorted
    (block, log_index) order, one buffer-flush at a time.

    Strategy: build a set of valid (tx_hash, log_index) keys from the index,
    then re-read the CSV filtering to those keys. Buffer up to BUFFER_BLOCKS
    distinct blocks before sorting and yielding, keeping RAM to ~2 chunks.
    """
    idx = pd.read_parquet(index_path, columns=["tx_hash", "log_index"])
    valid_keys = set(zip(idx["tx_hash"], idx["log_index"].astype(int)))
    del idx
    gc.collect()

    buffer: List[pd.DataFrame] = []
    buffered_blocks: Set = set()

    def _flush(buf: list) -> pd.DataFrame:
        return (
            pd.concat(buf, ignore_index=True)
              .sort_values(["block", "log_index"])
              .reset_index(drop=True)
        )

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, usecols=TARGET_COLS):
        chunk["dt_utc"]     = pd.to_datetime(chunk["timestamp"], utc=True)
        chunk["log_index"]  = chunk["log_index"].astype(int)

        # Filter to valid keys only
        active = chunk[
            chunk.apply(
                lambda r: (r["tx_hash"], int(r["log_index"])) in valid_keys,
                axis=1,
            )
        ]
        if active.empty:
            continue

        buffer.append(active)
        buffered_blocks.update(active["block"].unique())

        if len(buffered_blocks) >= BUFFER_BLOCKS:
            yield _flush(buffer)
            buffer = []
            buffered_blocks = set()
            gc.collect()

    if buffer:
        yield _flush(buffer)


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

def _process_chunk(
    df: pd.DataFrame,
    wallets: dict,
    cex_signals: Optional[pd.DataFrame],
) -> None:
    """
    Align CEX signals, then group by block and process trades.
    mark_to_market is fired once per wallet per block.
    """
    if cex_signals is not None and not cex_signals.empty:
        df       = align_ledger_to_cex(df, cex_signals)
        has_cex  = True
    else:
        df["price"] = np.nan
        df["bps"]   = np.nan
        has_cex     = False

    for _, block_df in df.groupby("block", sort=False):
        active_addrs: Set[str] = set()

        for row in block_df.itertuples(index=False):
            addr  = row.swap_sender
            price = float(row.execution_price)

            if addr not in wallets:
                wallets[addr] = WalletStats()

            wallets[addr].process_trade(
                side        = row.side,
                eth_qty     = float(row.base_qty_eth),
                usdc_qty    = float(row.quote_qty_usdc),
                log_index   = int(row.log_index),
                gas_fee_eth = float(row.tx_fee_eth),
                price       = price,
                cex_bps     = float(row.bps)   if has_cex else None,
                cex_price   = float(row.price) if has_cex else None,
            )
            active_addrs.add(addr)

        for addr in active_addrs:
            wallets[addr].mark_to_market(wallets[addr].last_price)


# ---------------------------------------------------------------------------
# Per-window processing  (runs against a pre-built shared index + CEX signals)
# ---------------------------------------------------------------------------

def _rank_and_save(df: pd.DataFrame, path: str, label: str) -> None:
    if df.empty:
        log.info("No wallets for %s.", label)
        return
    df = df.sort_values("net_usdc_after_gas", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df.to_csv(path, index=False)
    log.info("\u2705 %s \u2192 %s  (%d wallets)", label, path, len(df))


def _low_activity_path(output_file: str) -> str:
    """Derive low-activity filename from the main output path.
    e.g. leaderboard_static_daily.csv -> leaderboard_static_daily_low_activity.csv
    """
    base, ext = os.path.splitext(output_file)
    return f"{base}{LOW_ACTIVITY_SUFFIX}{ext}"


def _run_window(
    window_days: int,
    output_file: str,
    label: str,
    index_path: str,
    cex_signals: Optional[pd.DataFrame],
    final_price: Optional[float],
    now: datetime,
) -> None:
    """
    Process one rolling window using the pre-built sorted index.

    The index covers the longest window (7 days). For shorter windows we
    filter rows by timestamp after loading from the index — no re-scan of
    the CSV needed.

    `now` is passed in from the caller so all windows share the same
    reference point as the index build, avoiding any clock drift between calls.
    """
    cutoff = now - timedelta(days=window_days)
    log.info("--- Window: %s | %s -> %s ---",
             label,
             cutoff.strftime("%Y-%m-%dT%H:%MZ"),
             now.strftime("%Y-%m-%dT%H:%MZ"))

    wallets: Dict[str, WalletStats] = {}
    total_rows_in_window = 0

    for chunk in _stream_sorted_chunks(index_path):
        # Filter chunk to this window's cutoff
        window_chunk = chunk[chunk["dt_utc"] > cutoff]
        total_rows_in_window += len(window_chunk)
        if not window_chunk.empty:
            _process_chunk(window_chunk, wallets, cex_signals)
        del chunk, window_chunk
        gc.collect()

    log.info("%s: %d rows in window, %d wallets found.", label, total_rows_in_window, len(wallets))

    if not wallets:
        log.warning(
            "No trades found for %s window. "
            "Check that your CSV contains trades timestamped after %s.",
            label, cutoff.strftime("%Y-%m-%dT%H:%MZ"),
        )
        return

    log.info("%s: %d wallets processed.", label, len(wallets))

    # Final mark
    for wallet in wallets.values():
        mark = final_price if final_price else wallet.last_price
        if mark > 0:
            wallet.mark_to_market(mark)

    # Build results and split by MIN_TRADES
    results = [{"wallet": addr, **stats.finalize()} for addr, stats in wallets.items()]
    df_all   = pd.DataFrame(results)
    df_active = df_all[df_all["n_trades"] >= MIN_TRADES].copy()
    df_low    = df_all[df_all["n_trades"] <  MIN_TRADES].copy()

    _rank_and_save(df_active, output_file,                   f"{label} leaderboard")
    _rank_and_save(df_low,    _low_activity_path(output_file), f"{label} low-activity (<{MIN_TRADES} trades)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_leaderboard() -> None:
    now      = datetime.now(timezone.utc)
    # Widest cutoff covers all windows; CEX fetch and CSV index built once
    max_days = max(days for days, _, _ in WINDOWS)
    cutoff   = now - timedelta(days=max_days)
    kraken   = KrakenCEX()

    # 1. Fetch and resample CEX signals (shared across all windows)
    cex_signals, final_price = _build_cex_signals(kraken, cutoff)

    # 2. Pass 1 — build sorted/deduped index covering the full window span
    index_path = _build_sorted_index(cutoff)
    if index_path is None:
        log.info("Nothing to process.")
        return

    # 3. Run each window against the shared index
    try:
        for window_days, output_file, label in WINDOWS:
            _run_window(
                window_days  = window_days,
                output_file  = output_file,
                label        = label,
                index_path   = index_path,
                cex_signals  = cex_signals,
                final_price  = final_price,
                now          = now,
            )
    finally:
        os.unlink(index_path)   # always clean up temp index


if __name__ == "__main__":
    process_leaderboard()
