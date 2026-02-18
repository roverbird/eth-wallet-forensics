# 1. Raw Data

## 1.1 Source and Collection

Code: `collector.py`

Swap events emitted by the Uniswap v3 ETH/USDC 0.05% fee-tier pool
(`0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640`) on Ethereum mainnet
are collected using a custom JSON-RPC collector, `collector.py`
that processes blocks sequentially and caches transaction receipts 
per block to minimise RPC overhead.

Each record corresponds to one swap event and contains fifteen fields:
block number, log index, timestamp, three address fields, trade side,
base and quote quantities, execution price, price impact, gas used, gas
price, transaction fee, and transaction hash. The structure intentionally
mirrors proprietary trader logs used in CEX microstructure research,
enabling direct application of order-flow analysis methods to
decentralised venue data.

**Address decomposition.** Three distinct roles are captured per swap:
`tx_sender` (the account that signed the transaction), `swap_sender`
(the `msg.sender` of the pool's `swap()` call, typically a router), and
`recipient` (the address credited with output tokens). In direct swaps
all three coincide; in router-mediated swaps they diverge.

**Derived fields.** Execution price is decoded from the `sqrtPriceX96`
value in each Swap event log. Price impact (`impact_pct`) is computed
sequentially within each block—each trade's impact is measured against
the execution price of the immediately preceding swap in the same
block—consistent with the sandwich attack literature. The
`gas_price_gwei` field records `effectiveGasPrice` from the transaction
receipt; under EIP-1559 this is the priority tip, not base fee plus tip.
This is confirmed empirically: 61.3% of transactions record sub-1-Gwei
values, inconsistent with mainnet base fees during the period (1–20
Gwei). Gas fields are therefore treated as signals of inclusion urgency
rather than absolute cost.

# 2. Processed Data

## 2.1 Unit of Analysis and Sample Restriction

The unit of analysis is the individual recipient wallet.

Code: `wrapper.py`

## 2.2 Feature Engineering

`wrapper.py` computes 26 per-wallet features grouped into five categories:
address-based, temporal, volume, block-level, and CEX-derived. Where
features overlap with those in Niedermayer et al. (2024), we follow
their definitions; features specific to the on-chain builder detection
context are defined below.

### 2.2.1 Address-Based Features

Following Niedermayer et al. (2024), **`n_leading_zeros`** counts
leading zeros in the wallet's 40-character hexadecimal address.
Addresses with more leading zeros are typically mined deliberately to
reduce gas costs in smart contract interactions, and serve as a proxy
for sophisticated, contract-based actors.

### 2.2.2 Temporal Features

**`sleepiness_hr`** is the maximum gap between consecutive trades, in
hours. This departs from the interval-averaged formulation of
Niedermayer et al. (2024), who average maximum gaps across two-day
windows; we take the single global maximum over the observation period.
A high value indicates a wallet that goes dormant for extended periods
between activity bursts — characteristic of event-driven bots — while a
low value indicates continuous market monitoring.

**`sender_diversity`** is the ratio of unique `tx_sender` addresses to
total trades. A value near zero means all trades were routed through a
single caller (proprietary infrastructure); a value near one means each
trade came from a distinct sender (typical of retail users routing
through public interfaces).

**`self_swap_ratio`** is the fraction of trades where `tx_sender`
equals `recipient` — that is, the wallet submitted and received its own
swap without an intermediary router. This is a binary indicator of
direct, self-initiated execution. A ratio of 1.0 combined with a high `avg_log_index` 
is the pathognomonic sign of synthetic volume. While self-swapping is 
often ignored in MEV literature, in our sample it serves as a filter 
for wash traders who inflate multichain valuations while operating at 
the lowest-priority tiers of the block (median log index > 300)."

### 2.2.3 Volume and Price Features

**`value_clustering_score`** is the fraction of ETH trade sizes that
qualify as round numbers, defined as values whose string representation
has fewer than four significant decimal digits. Following Niedermayer et
al. (2024), round-number clustering reflects
human cognitive reference points; its absence in a bot wallet is
expected and its presence may indicate wash trading or manual
intervention.

**`mean_base_qty_eth`**, **`mean_quote_qty_usdc`**,
**`mean_gas_price_gwei`**, and **`mean_tx_fee_eth`** are arithmetic
means of per-trade quantities. Gas price is interpreted as an inclusion
urgency signal rather than an absolute cost measure, consistent with the
EIP-1559 recording issue described in Section 3.2.

**`net_usdc_flow`** is the signed sum of USDC flows over the observation
window:

$$
\text{net\_usdc\_flow}(w) = \sum_{t} \text{flow}_t, \quad
\text{where} \quad
\text{flow}_t =
\begin{cases}
+\text{usdc}_t & \text{if SELL\_ETH} \\
-\text{usdc}_t & \text{if BUY\_ETH}
\end{cases}
$$

A positive value indicates net USDC extraction from the pool (wallet is
a net seller of ETH); a negative value indicates net USDC injection
(wallet is a net buyer). This is the primary profitability indicator.

### 2.2.4 Block-Level Features

Four features capture intra-block positioning and dominance, computed
from the pool's transaction log indexed by block number and log index.

**`avg_log_index`** is the mean position of the wallet's transactions
within their respective blocks. Lower values indicate earlier placement,
which in the context of a Uniswap pool is a direct signal of block
construction access: randomly submitted transactions land at positions
determined by mempool ordering, while block builders can place their own
transactions first.

**`block_capture_rate`** is the fraction of blocks in which the wallet
accounts for more than 50% of the pool's total swap volume. A wallet
achieving majority volume share in a block has effectively dominated
that block's price discovery.

**`avg_block_share`** is the mean fraction of per-block pool volume
attributable to the wallet, across all blocks in which it appears.

**`avg_txs_per_block`** and **`multi_tx_rate`** measure the intensity
and prevalence of multi-transaction execution within single blocks.
`avg_txs_per_block` is the mean number of swaps placed by the wallet in
a block; `multi_tx_rate` is the fraction of blocks containing more than
one such swap. Values above 2 on `avg_txs_per_block` are consistent
with atomic sandwich execution (frontrun + backrun bracketing a victim
trade).

### 2.2.5 CEX-Derived Feature

Code: `cex_library.py`

**`alpha_reaction_rate`** measures directional alignment between the
wallet's DEX trades and concurrent Kraken price movements. Tick-level
trade data for the XETHZUSD pair is fetched from the Kraken REST API,
resampled to one-second intervals, and forward-filled with a 60-second
cap to avoid attaching stale prices to DEX events. Price returns are
expressed in basis points. A CEX signal is defined as a one-second
interval with $|\text{bps}| \geq 5$. For each DEX swap occurring within
a signal window, a reaction is recorded if the trade direction matches
the CEX price movement:

$$
\text{reaction}_t = \mathbf{1}\!\left[
  (\text{bps}_t > 0 \wedge \text{side}_t = \text{BUY\_ETH}) \;\vee\;
  (\text{bps}_t < 0 \wedge \text{side}_t = \text{SELL\_ETH})
\right]
$$

The per-wallet rate is the mean over all signal-adjacent trades. A high
rate indicates that a wallet consistently trades in the direction of
concurrent CEX price moves, consistent with cross-venue arbitrage or
CEX-informed order placement.

## 2.3 Coordination Detection (WalletForensics)

Code: `wallet_forensics.py`

Two features — `coordination_score` and `cluster_size` — are produced
by a bespoke forensic module (`WalletForensics`) that detects
statistically non-random co-occurrence among wallet addresses in the
swap stream. The module requires no external data and operates entirely
on the transaction log.

The raw stream is converted into two sets of wallet co-occurrence
sequences: **atomic** (wallets sharing a `tx_hash`) and **temporal**
(wallets sharing a timestamp). For atomic groupings, all distinct
wallets within a transaction are retained in first-seen order; for
temporal groupings, consecutive duplicate wallets are collapsed to focus
on meaningful co-occurrence. Sequences containing fewer than two
distinct wallets are discarded.

Bigram and trigram collocation statistics are computed over the resulting
document set using NLTK's likelihood-ratio measure, with a minimum
frequency filter of 2. Wallet pairs and triplets whose co-occurrence
frequency significantly exceeds the independence baseline are flagged as
syndicate candidates:

$$
\mathcal{S} = \{ w \mid w \text{ appears in top-ranked collocations} \}
$$

A swap event is tagged as coordinated if its wallet belongs to
$\mathcal{S}$ and the same event contains at least one other syndicate
member:

$$
\text{is\_coordinated}(w, t) =
\mathbf{1}\!\left[
  w \in \mathcal{S} \;\wedge\; \left|\{w' \in \text{tx}(t) \cap \mathcal{S}\}\right| \geq 2
\right]
$$

The **`coordination_score`** is the percentage of a wallet's trades
that satisfy this condition. The **`cluster_size`** counts the number of
distinct stable partners — wallets sharing at least two co-occurrence
contexts with the focal wallet:

$$
\text{cluster\_size}(w) = \left|\{ w' \mid |C_{ww'}| \geq 2 \}\right|
$$

where $C_{ww'}$ is the set of contexts in which both $w$ and $w'$
appear.

## 2.4 Negative Binomial Modeling of Epoch Activity

### 2.4.1 Epoch Construction

Each block is assigned to an epoch of size 32 blocks (one Ethereum
proof-of-stake epoch, approximately 6.4 minutes). For each wallet, we
construct a count vector $X_i = \{x_{i1}, \ldots, x_{iT}\}$ where
$x_{ie}$ is the number of swaps executed in epoch $e$. The vector spans
all epochs present in the dataset, so epochs where the wallet did not
trade contribute zeros.

Before fitting, the vector is cropped to the wallet's **active
lifespan**: leading and trailing zero epochs that precede the wallet's
first swap and follow its last are removed. This prevents the global
epoch range — which is shared across all wallets regardless of when they
entered the pool — from suppressing the variance estimate and biasing
the MLE toward the Poisson distribution. At least three non-zero epochs
within the cropped window are required; wallets with fewer active epochs
receive `nb_status = insufficient_active_epochs` and NaN parameter
values.

The choice of epoch rather than hour as the temporal unit is motivated
by our finding that 32-block windows more closely match the natural
timescale of burst activity (sandwich sequences, arbitrage runs) than
60-minute bins, which aggregate across structurally distinct market
conditions.

### 2.4.2 Distributional Model

We model the cropped epoch count sequences using the Negative Binomial
distribution:

$$
X \sim \mathrm{NB}(k,\, p),
\qquad
\mu = k\frac{1-p}{p},
\qquad
\mathrm{Var}[X] = k\frac{1-p}{p^2}
$$

The NBD is appropriate for count data exhibiting overdispersion
(variance exceeding the mean), a necessary condition verified via the
Fano factor $F = \hat{\sigma}^2 / \hat{\mu}$. Sequences with $F \leq 1$
are classified as underdispersed and excluded from fitting.

The NBD has well-known limiting cases in its shape parameter $k$: the
geometric distribution at $k = 1$, the logarithmic series distribution
as $k \to 0^+$, and the Poisson distribution as $k \to \infty$. The
success probability $p \in (0, 1)$ is bounded by construction, making it
a normalised per-epoch regularity measure directly suitable as a feature.

### 2.4.3 Estimation

Parameters are estimated by maximum likelihood using
`statsmodels.discrete.NegativeBinomial` with an intercept-only design
matrix (100 maximum iterations). The fitted model returns the
expected count $\hat{\mu} = \exp(\hat{\beta}_0)$ and the NB
overdispersion parameter $\hat{\alpha}$, from which standard $(k, p)$
parameters are recovered as:

$$
\hat{p} = \frac{1}{1 + \hat{\mu}\hat{\alpha}},
\qquad
\hat{k} = \frac{\hat{\mu}\hat{p}}{1 - \hat{p}}
$$

Parameter validity is checked post-estimation: fits returning
$\hat{\alpha} \leq 0$, $\hat{\mu} \leq 0$, $\hat{p} \notin (0,1)$, or
$\hat{k} \leq 0$ are rejected and recorded as `invalid_params`. Wllets that do not
yield a valid MLE solution receive NaN values for $k$ and $p$, so
that downstream correlations use only well-identified parameters.

**Interpretation.** A low $p$ with high Fano factor indicates
concentrated burst trading — few epochs with high activity interspersed
with long inactivity — consistent with event-driven or sandwich-style
execution. A high $p$ with Fano near 1 indicates evenly distributed
activity across epochs, consistent with scheduled or liquidity-provision
behaviour. The $k$ parameter captures the concentration of bursts: high
$k$ implies that activity, when it occurs, is distributed across many
moderate-sized epochs; low $k$ implies fewer, more extreme spikes.

## 2.5 Etherscan Label Alignment and On-Chain Profiling

`align_profile.py`

The per-wallet feature dataset is enriched in two further steps before
analysis: label alignment against a curated MEV bot registry, and
direct on-chain profiling of each wallet address.

**Label alignment.** Recipient wallet addresses are cross-referenced
against a manually curated list of Etherscan-labelled MEV bots
(`etherscan_mev_list.csv`). Matching is performed case-insensitively on
the full 42-character hex address. Wallets appearing in the registry
receive a `name_label` of `mev_bot`; all remaining wallets are assigned
`unknown`.

**On-chain profiling.** For each wallet, three fields are retrieved
directly from the Ethereum node via JSON-RPC, batched asynchronously
through the Alchemy endpoint to minimise latency:

- **`bytecode_len`**: the length in bytes of the deployed contract
  bytecode at the address, obtained via `eth_getCode`. A non-zero value
  confirms the address is a smart contract rather than an externally
  owned account (EOA). Contract bytecode length is a feature used by
  Niedermayer et al. (2024) as a proxy for architectural complexity;
  sophisticated MEV bots typically deploy purpose-built contracts
  optimised for gas efficiency, resulting in compact but non-trivial
  bytecode.

- **`eth_balance`**: the current ETH balance of the address in wei,
  obtained via `eth_getBalance`. Balance provides a coarse indicator of
  financial capacity and operational status; a wallet with a near-zero
  balance is either dormant or routes capital through a separate funding
  address.

- **`is_contract`**: a boolean derived from `bytecode_len > 0`,
  distinguishing smart contract wallets from EOAs. In our sample,
  contract wallets account for the majority of identified MEV bots, as
  on-chain execution logic is required to implement sandwich attacks and
  atomic arbitrage within a single transaction.

- **`is_active_mev_proxy`**: a boolean flag derived from heuristics
  applied to the bytecode and transaction patterns, indicating whether
  the contract exhibits structural signatures consistent with a MEV
  proxy — a thin routing contract that delegates execution to a separate
  implementation contract. Proxy architectures are commonly used to
  separate upgradeable strategy logic from the capital-holding address.

These four fields are appended to the per-wallet feature dataset,
producing the final analytical table of 30 variables used in all
subsequent analysis. The label column enables stratified comparison
between Etherscan-confirmed bots and the unlabelled population, while
the on-chain profile fields provide infrastructure-level covariates that
are orthogonal to the swap-log-derived behavioural features.
