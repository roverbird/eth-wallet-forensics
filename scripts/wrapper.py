import sys
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from scipy import stats as scipy_stats
from statsmodels.discrete import discrete_model
from wallet_forensics import WalletForensics
from cex_library import KrakenCEX, align_ledger_to_cex

# =====================================================
# CONFIGURATION
# =====================================================
STARTING_DATE = "2026-02-01"  # Earliest date to fetch CEX data from Kraken
FANO_THRESHOLD = 1.0          # Fano = var/mean; values < 1.0 indicate Poisson (no overdispersion), NB not appropriate
MIN_OBSERVATIONS = 3          # Minimum number of active epochs (non-zero) required for a valid NB fit
NB_MAX_ITER = 100             # Maximum MLE iterations before giving up on NB convergence
WINDOW_SIZE = 32              # Ethereum epoch = 32 blocks (~6.4 minutes); temporal unit for NB count vector

# =====================================================
# HELPERS
# =====================================================

def count_leading_zeros(address):
    # Vanity address indicator: more leading zeros = likely deliberately mined address (MEV bot signal)
    addr_hex = address[2:] if address.startswith('0x') else address
    return len(addr_hex) - len(addr_hex.lstrip('0'))

def is_round_number(val):
    # Returns True if trade size looks like a human-chosen round number (e.g. 1.0, 0.5)
    # High frequency of round numbers = possible wash trading or manual intervention
    if val == 0: return True
    val_str = str(val)
    return val_str.rstrip('0').endswith('.') or len(val_str.split('.')[-1]) < 4

def calculate_sleepiness(timestamps):
    # Maximum gap between consecutive trades in hours
    # High value = bot goes dormant for long periods (scheduled or event-driven)
    # Low value = bot monitors continuously
    if len(timestamps) < 2: return 0.0
    ts_sorted = sorted(timestamps)
    deltas = [ts_sorted[i] - ts_sorted[i-1] for i in range(1, len(ts_sorted))]
    return max(deltas) / 3600.0

def estimate_nb_parameters(values):
    """
    Fits a Negative Binomial distribution to the wallet's per-epoch trade counts.

    Crops the count vector to the wallet's active lifespan (first to last non-zero epoch)
    before fitting, so leading/trailing zeros from the global epoch range don't suppress
    the variance estimate and bias the MLE toward Poisson.

    Returns:
        k     - NB dispersion (concentration of bursts; higher = fewer, larger bursts)
        p     - NB success probability (regularity; higher p = more evenly spaced activity)
        fano  - Variance/mean of the cropped window (overdispersion diagnostic)
        status - Fit outcome string
    """
    if len(values) == 0 or np.all(values == 0):
        return np.nan, np.nan, 0.0, "all_zeros"

    # Crop to active window: remove leading/trailing zero epochs that don't belong to this wallet
    indices = np.nonzero(values)[0]
    first_idx, last_idx = indices[0], indices[-1]
    cropped_values = values[first_idx : last_idx + 1]

    # Require at least MIN_OBSERVATIONS non-zero epochs within the active window
    if len(indices) < MIN_OBSERVATIONS:
        return np.nan, np.nan, 0.0, f"insufficient_active_epochs (n={len(indices)})"

    mean_v = np.mean(cropped_values)
    var_v  = np.var(cropped_values)
    fano   = var_v / mean_v if mean_v > 0 else 0.0

    # NB requires overdispersion; Poisson or underdispersed data should not be fitted
    if fano < FANO_THRESHOLD:
        return np.nan, np.nan, fano, "underdispersed"

    try:
        exog   = np.ones((len(cropped_values), 1))  # intercept-only model
        model  = discrete_model.NegativeBinomial(cropped_values, exog)
        result = model.fit(disp=False, maxiter=NB_MAX_ITER, warn_convergence=False)

        mu    = np.exp(result.params[0])   # expected count per epoch (exponentiated intercept)
        alpha = result.params[1]           # NB overdispersion parameter from statsmodels

        if alpha <= 0 or mu <= 0:
            return np.nan, np.nan, fano, "invalid_params"

        # Convert statsmodels (mu, alpha) to standard (k, p) parameterisation
        # NB(k, p): mean = k(1-p)/p,  variance = k(1-p)/p^2
        p = 1.0 / (1.0 + mu * alpha)
        k = mu * p / (1.0 - p)

        if not (0 < p < 1) or k <= 0:
            return np.nan, np.nan, fano, "invalid_params"

        return float(k), float(p), fano, "success"
    except:
        return np.nan, np.nan, fano, "fit_error"


# =====================================================
# MAIN PIPELINE
# =====================================================
if len(sys.argv) != 5:
    print("Usage: python stat6.py input_ledger.csv min_freq max_freq result.csv")
    sys.exit(1)

csv_input, min_freq_str, max_freq_str, result_file = sys.argv[1:]
min_freq, max_freq = int(min_freq_str), int(max_freq_str)

# ------------------------------------------------------------------
# STEP 0: CEX SIGNAL ACQUISITION
# ------------------------------------------------------------------
print(f"📡 STEP 0: Syncing CEX Signals...")
ledger_raw = pd.read_csv(csv_input)
ledger_raw['dt_utc'] = pd.to_datetime(ledger_raw['timestamp'], utc=True)

ledger_raw['block']    = ledger_raw['block'].astype(int)
ledger_raw['epoch_id'] = ledger_raw['block'] // WINDOW_SIZE  # integer epoch index per row

global_start = datetime.fromisoformat(STARTING_DATE).timestamp()
data_end     = ledger_raw['dt_utc'].max().timestamp()

try:
    kraken      = KrakenCEX()
    cex_ticks   = kraken.fetch_tick_history(pair="XETHZUSD", start_ts=global_start, end_ts=data_end)
    if cex_ticks.empty:
        raise ValueError("Empty CEX response")
    market_signals = kraken.get_market_signals(cex_ticks, bps_threshold=5)
    print(f"   ✓ {len(cex_ticks):,} ticks fetched, {len(market_signals):,} signals (|bps| ≥ 5)")
except Exception as e:
    print(f"⚠️  CEX fetch failed: {e} — continuing without market signals")
    market_signals = pd.DataFrame(columns=['price', 'bps'])

# ------------------------------------------------------------------
# STEP 1: FORENSIC PATTERN ANALYSIS
# ------------------------------------------------------------------
print("🧬 STEP 1: Running WalletForensics...")
forensics = WalletForensics(csv_input, min_trades=min_freq)
f_stats   = forensics.run_analysis()

# Enforce both frequency bounds (forensics only applies min_trades internally)
wallet_trade_counts = f_stats['total_trades']
f_stats = f_stats[
    (wallet_trade_counts >= min_freq) &
    (wallet_trade_counts <= max_freq)
]
candidate_wallets = set(f_stats.index)

# ------------------------------------------------------------------
# STEP 2: DATA COLLECTION (EPOCH-BASED)
# ------------------------------------------------------------------
print("🕵️ STEP 2: Processing Ledger with Epoch Windows...")
if market_signals.empty:
    aligned_ledger          = ledger_raw.copy()
    aligned_ledger['bps']   = 0.0
    aligned_ledger['price'] = np.nan
else:
    aligned_ledger = align_ledger_to_cex(ledger_raw, market_signals)

# epoch_id is computed on ledger_raw; ensure it survives the merge_asof
if 'epoch_id' not in aligned_ledger.columns:
    aligned_ledger['epoch_id'] = aligned_ledger['block'].astype(int) // WINDOW_SIZE

# Accumulators (keyed by wallet address)
epoch_activity  = defaultdict(lambda: defaultdict(int))  # epoch_id → wallet → trade count
wallet_senders  = defaultdict(set)    # unique tx_sender addresses seen per wallet
wallet_self_calls = defaultdict(int)  # trades where tx_sender == recipient (contract self-call)
wallet_net_usdc = defaultdict(float)  # cumulative PnL: positive = net received USDC
wallet_timestamps = defaultdict(list) # unix timestamps of every trade
wallet_counts   = defaultdict(int)    # total trade count
wallet_log_indices = defaultdict(list)# position of each tx within its block (log_index)
wallet_numeric  = defaultdict(lambda: defaultdict(list))  # raw numeric field values per trade
wallet_alpha_stats = defaultdict(list)  # 1/0 per trade: did DEX direction match CEX signal?

for _, row in aligned_ledger.iterrows():
    recip = row.get("recipient")
    if not recip or recip not in candidate_wallets:
        continue

    eid = row['epoch_id']
    epoch_activity[eid][recip] += 1
    wallet_counts[recip] += 1

    sender = row.get("tx_sender")
    if sender:
        wallet_senders[recip].add(sender)
        # NOTE: this detects tx_sender == recipient (self-call), not swap_sender == recipient (self-swap)
        # Both are MEV-relevant but measure different things; column is named self_swap_ratio for legacy reasons
        if str(sender).lower() == str(recip).lower():
            wallet_self_calls[recip] += 1

    wallet_timestamps[recip].append(row['dt_utc'].timestamp())
    wallet_log_indices[recip].append(float(row.get("log_index", 0)))

    wallet_numeric[recip]["base_qty_eth"].append(float(row.get("base_qty_eth", 0)))
    wallet_numeric[recip]["quote_qty_usdc"].append(float(row.get("quote_qty_usdc", 0)))
    wallet_numeric[recip]["gas_price_gwei"].append(float(row.get("gas_price_gwei", 0)))
    wallet_numeric[recip]["tx_fee_eth"].append(float(row.get("tx_fee_eth", 0)))

    # Alpha reaction: 1 if DEX trade direction matches CEX price move direction, else 0
    # Only scored when a significant CEX move (|bps| >= 5) is attached to this trade
    bps  = row.get('bps', 0) or 0  # guard against NaN from failed CEX fetch
    side = row.get('side')
    if abs(bps) >= 5:
        is_reaction = (bps > 0 and side == "BUY_ETH") or (bps < 0 and side == "SELL_ETH")
        wallet_alpha_stats[recip].append(1 if is_reaction else 0)

    # PnL: SELL_ETH = receive USDC (positive), BUY_ETH = spend USDC (negative)
    usdc = float(row.get("quote_qty_usdc", 0))
    flow = -usdc if row.get("side") == "BUY_ETH" else usdc
    wallet_net_usdc[recip] += flow

# ------------------------------------------------------------------
# STEP 3: CONSOLIDATE & EXPORT
# ------------------------------------------------------------------
print(f"💾 STEP 3: Writing Master Report (Epoch Granularity)...")
all_epochs = sorted(epoch_activity.keys())

# Output columns (26 total — must match writer.writerow exactly)
header = [
    "wallet",               # Ethereum address of the recipient wallet
    "total_trades",         # Total swap count over the full observation window
    "k",                    # NB dispersion: higher = fewer, larger bursts; lower = many small bursts
    "p",                    # NB success prob: higher = regular spacing; lower = bursty/clustered
    "fano",                 # Variance/mean of per-epoch counts; > 1 confirms overdispersion
    "nb_status",            # Fit outcome: success / underdispersed / insufficient_active_epochs / fit_error / all_zeros
    "unique_senders",       # Count of distinct tx_sender addresses that routed through this recipient
    "sender_diversity",     # unique_senders / total_trades: how varied the routing is (0=single router, 1=every tx different)
    "self_swap_ratio",      # Fraction of trades where tx_sender == recipient (self-initiated; see note above)
    "n_leading_zeros",      # Leading zeros in hex address: proxy for vanity/mined address (higher = more deliberate)
    "sleepiness_hr",        # Longest gap between consecutive trades in hours (high = dormant periods)
    "value_clustering_score", # Fraction of ETH trade sizes that are round numbers (high = possible wash trading)
    "avg_log_index",        # Mean position of wallet's txs within their blocks (lower = earlier = builder signal)
    "coordination_score",   # % of trades that co-occur with syndicate partners in same tx/timestamp (forensics output)
    "cluster_size",         # Number of stable co-trading partners detected by forensics
    "block_capture_rate",   # % of blocks where this wallet accounts for >50% of pool volume (dominant builder signal)
    "avg_block_share",      # Mean fraction of pool volume this wallet contributes per block it appears in
    "avg_txs_per_block",    # Average number of swaps this wallet places in a single block (>2 = sandwich pattern)
    "multi_tx_rate",        # Fraction of blocks where wallet placed more than one swap (atomic multi-step execution)
    "mean_base_qty_eth",    # Mean ETH size per swap
    "mean_quote_qty_usdc",  # Mean USDC size per swap
    "mean_gas_price_gwei",  # Mean gas price paid (higher = willing to overpay for inclusion priority)
    "mean_tx_fee_eth",      # Mean total gas cost per transaction in ETH
    "net_usdc_flow",        # Total PnL in USDC: positive = net seller/profitable, negative = net buyer/unprofitable
    "alpha_reaction_rate",  # Fraction of CEX-signal-adjacent trades where DEX direction matched CEX move
    "role",                 # Classification: Profitable, or Unprofitable
]

with open(result_file, "w", newline="", encoding="utf-8") as res:
    writer = csv.writer(res)
    writer.writerow(header)

    for w in sorted(candidate_wallets):
        # Build per-epoch count vector spanning the full dataset epoch range
        # Zeros included for epochs where this wallet was inactive (needed for NB variance)
        vals = np.array([epoch_activity[e][w] for e in all_epochs])

        k, p, fano, status = estimate_nb_parameters(vals)
        total = wallet_counts[w]

        # Behaviorals
        n_zeros    = count_leading_zeros(w)
        sleepy     = calculate_sleepiness(wallet_timestamps[w])
        eth_vals   = wallet_numeric[w]["base_qty_eth"]
        val_cluster = sum(1 for v in eth_vals if is_round_number(v)) / len(eth_vals) if eth_vals else 0.0

        # Means
        m_eth  = np.mean(wallet_numeric[w]["base_qty_eth"])
        m_usdc = np.mean(wallet_numeric[w]["quote_qty_usdc"])
        m_gas  = np.mean(wallet_numeric[w]["gas_price_gwei"])
        m_fee  = np.mean(wallet_numeric[w]["tx_fee_eth"])

        # Alpha reaction rate (0.0 if no CEX signals were attached to this wallet's trades)
        reactions  = wallet_alpha_stats[w]
        alpha_rate = np.mean(reactions) if reactions else 0.0

        # Forensics outputs — use `or 0` to guard against None when column missing from f_stats
        row_f       = f_stats.loc[w]
        capture_rate = float(row_f.get('block_capture_rate', 0) or 0)
        avg_block_shr = float(row_f.get('avg_block_share', 0) or 0)
        avg_txs_blk  = float(row_f.get('avg_txs_per_block', 0) or 0)
        multi_tx     = float(row_f.get('multi_tx_rate', 0) or 0)
        coord_score  = float(row_f.get('coordination_score', 0) or 0)
        cluster_sz   = int(row_f.get('cluster_size', 0) or 0)

        net_flow = wallet_net_usdc[w]

        # Role classification
        if net_flow >= 0:
            role = "Profitable"     # Net USDC receiver over observation window
        else:
            role = "Unprofitable"   # Net USDC spender over observation window

        writer.writerow([
            w, total, f"{k:.4f}", f"{p:.4f}", round(fano, 3), status,
            len(wallet_senders[w]), round(len(wallet_senders[w]) / total, 3),
            round(wallet_self_calls[w] / total, 3),
            n_zeros, round(sleepy, 2), round(val_cluster, 3),
            round(np.mean(wallet_log_indices[w]), 2),
            coord_score, cluster_sz,
            capture_rate, avg_block_shr, avg_txs_blk, multi_tx,
            round(m_eth, 6), round(m_usdc, 2), round(m_gas, 2), round(m_fee, 6),
            round(net_flow, 2), round(alpha_rate, 3),
            role,  # BUG FIX: was missing from writer.writerow, causing header/data column misalignment
        ])

print("✅ DONE: Grouped by block Epochs.")
