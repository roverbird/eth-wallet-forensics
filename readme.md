# Ethereum Wallet Forensics Kit for Behavioral Profiling

This repository contains the analytical pipeline and data collection suite used to identify and categorize Ethereum market participants in Uniswap v3 liquidity pools. Built for our MEV bot research project.

## Research Context

Eallet labeling from Etherscan and other systems often lags behind the MEV industry. This project introduces a novel Negative Binomial Distribution (NBD) and NLP analytics methodology to profile ETH addresses.

## Core Pipeline Components

The codebase is organized into a modular pipeline:

### 1. Data Acquisition

* **`scripts/collector.py`**: A high-performance Uniswap Pool monitoring system. It processes blocks sequentially to generate a transaction log, including raw event logs and transaction input data, similar to propriatory CEX trader login logs.

### 2. Forensic Analysis

* **`scripts/wrapper.py`**: The primary execution engine that transforms raw logs into the processed feature set used for behavioral taxonomy.
* **`scripts/wallet_forensics.py`**: The **WalletForensics** module. It uses NLP-inspired collocation analysis (bigrams/trigrams) to detect statistically non-random co-occurrence of addresses, identifying coordinated ETH wallet clusters and syndicates.
* **`scripts/cex_library.py`**: Utilities for fetching tick-level price data from CEX (e.g., Kraken). This enables the calculation of `alpha_reaction_rate` by aligning DEX swaps with external market signals.

### 3. Profile Enrichment

* **`scripts/align_profile.py`**: Enriches behavioral data with infrastructure-level signals, including `bytecode_len` (contract complexity), and `eth_balance`.
* **`scripts/alchemy_lib.py`**: A dedicated wrapper for the Alchemy JSON-RPC API to handle high-concurrency on-chain data requests.

## License

This project is licensed under the **MIT License**.

## Data Availability

The underlying dataset (cleaned Uniswap v3 ETH/USDC swap logs) used in the associated research paper is archived on **Zenodo** at: `[ZENODO DOI LINK HERE]`

