# wallet_forensics library

import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.collocations import (
    BigramAssocMeasures,
    TrigramAssocMeasures,
    BigramCollocationFinder,
    TrigramCollocationFinder,
)

class WalletForensics:
    def __init__(self, filepath, min_trades=5, top_n=200):
        # Load and clean specific columns from your raw input
        self.df = pd.read_csv(filepath)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values(["timestamp"]).reset_index(drop=True)
        
        self.min_trades = min_trades
        self.top_n = top_n
        self.syndicate_members = set()
        self.shared_events_counter = defaultdict(set)

    def _build_documents(self, grouping_col):
        """Groups addresses by tx_hash or timestamp to find sequences of co-occurrence."""
        documents = []
        for key, g in self.df.groupby(grouping_col, sort=False):
            wallets = g["recipient"].dropna().astype(str).tolist()
            if not wallets:
                continue
            
            if grouping_col == "tx_hash":
                # For atomic tx grouping: keep all unique wallets (order irrelevant)
                # A single tx may have 2+ different recipients — that IS coordination
                seq = list(dict.fromkeys(wallets))  # unique, preserve first-seen order
            else:
                # For temporal grouping: collapse consecutive duplicates
                # Same wallet firing multiple times in same second is noise
                seq = [wallets[0]]
                for w in wallets[1:]:
                    if w != seq[-1]:
                        seq.append(w)
            
            if len(set(seq)) >= 2:
                documents.append((key, seq))
        return documents

    def _extract_patterns(self, documents):
        """Uses NLTK Likelihood Ratio to find non-random wallet pairings."""
        found_members = set()
        doc_sequences = [seq for _, seq in documents]
        if not doc_sequences:
            return found_members
        
        # 1. Bigram Analysis
        b_finder = BigramCollocationFinder.from_documents(doc_sequences)
        b_finder.apply_freq_filter(2) 
        
        for w1, w2 in b_finder.nbest(BigramAssocMeasures.likelihood_ratio, self.top_n):
            found_members.update((w1, w2))
            pair = tuple(sorted((w1, w2)))
            for ctx, seq in documents:
                if w1 in seq and w2 in seq:
                    self.shared_events_counter[pair].add(ctx)

        # 2. Trigram Analysis
        t_finder = TrigramCollocationFinder.from_documents(doc_sequences)
        t_finder.apply_freq_filter(2)
        
        for w1, w2, w3 in t_finder.nbest(TrigramAssocMeasures.likelihood_ratio, self.top_n):
            found_members.update((w1, w2, w3))
            for pair in [(w1, w2), (w2, w3), (w1, w3)]:
                p = tuple(sorted(pair))
                for ctx, seq in documents:
                    if pair[0] in seq and pair[1] in seq:
                        self.shared_events_counter[p].add(ctx)
        return found_members

    def run_analysis(self):
            # Step 0: Identify Direct Swaps (Pre-processing)
            if 'swap_sender' in self.df.columns:
                self.df["is_direct"] = (self.df["recipient"] == self.df["swap_sender"]).astype(int)
            else:
                self.df["is_direct"] = 0

            # Step 1: Pattern Extraction
            atomic_docs = self._build_documents("tx_hash")
            temporal_docs = self._build_documents("timestamp")
            self.syndicate_members = self._extract_patterns(atomic_docs) | self._extract_patterns(temporal_docs)

            # Step 2: Double-Lock Tagging
            dirty_txs = {tx for tx, g in self.df.groupby("tx_hash") 
                         if len([w for w in g["recipient"].unique() if str(w) in self.syndicate_members]) >= 2}
            
            self.df["is_coordinated"] = (
                self.df["recipient"].astype(str).isin(self.syndicate_members) & 
                self.df["tx_hash"].isin(dirty_txs)
            )

            # Step 3: Aggregation (Including Direct Swap Ratio)
            stats = self.df.groupby("recipient").agg(
                total_trades=("recipient", "count"),
                coordinated_trades=("is_coordinated", "sum"),
                direct_swaps=("is_direct", "sum")
            )

            stats["coordination_score"] = (stats["coordinated_trades"] / stats["total_trades"] * 100).round(2)
            stats["direct_swap_ratio"] = (stats["direct_swaps"] / stats["total_trades"]).round(4)

            # Step 4: Cluster Scaling
            stable_partners = defaultdict(set)
            for (w1, w2), contexts in self.shared_events_counter.items():
                if len(contexts) >= 2:
                    stable_partners[w1].add(w2)
                    stable_partners[w2].add(w1)

            stats["cluster_size"] = stats.index.map(lambda w: len(stable_partners.get(str(w), ())))

            # Step 5: Calculate Block Metrics
            block_metrics = self._calculate_block_metrics()
            
            if block_metrics:
                for metric_name in ['block_capture_rate', 'avg_block_share', 'avg_txs_per_block', 'multi_tx_rate']:
                    stats[metric_name] = stats.index.map(
                        lambda w: block_metrics.get(str(w), {}).get(metric_name, 0.0)
                    )

            return stats[stats["total_trades"] >= self.min_trades].sort_values("coordination_score", ascending=False)
    
    def _calculate_block_metrics(self):
        """
        Calculate block-level dominance metrics for builder detection.
        
        Returns dictionary with per-wallet block capture metrics:
        - block_capture_rate: % of blocks where wallet has >50% of volume
        - avg_block_share: Average % of block volume
        - avg_txs_per_block: Average transactions per block
        - multi_tx_rate: % of blocks with multiple transactions
        """
        # Check if block column exists
        if 'block' not in self.df.columns:
            print("⚠️  No 'block' column found, skipping block metrics")
            return {}
        
        # Check if volume column exists
        volume_col = None
        for col in ['quote_qty_usdc', 'base_qty_eth', 'volume']:
            if col in self.df.columns:
                volume_col = col
                break
        
        if volume_col is None:
            print("⚠️  No volume column found, skipping block metrics")
            return {}
        
        print(f"📊 Calculating block capture metrics using '{volume_col}'...")
        
        # Calculate total volume per block
        block_totals = self.df.groupby('block').agg({
            volume_col: lambda x: abs(x).sum()
        }).reset_index()
        block_totals.columns = ['block', 'block_total_volume']
        
        # Calculate per-wallet, per-block volume
        wallet_block_volume = self.df.groupby(['recipient', 'block']).agg({
            volume_col: lambda x: abs(x).sum()
        }).reset_index()
        wallet_block_volume.columns = ['wallet', 'block', 'wallet_volume']
        
        # Merge to get shares
        merged = wallet_block_volume.merge(block_totals, on='block')
        merged['block_share'] = merged['wallet_volume'] / merged['block_total_volume']
        
        # Transaction counts per block
        txs_per_block = self.df.groupby(['recipient', 'block']).size().reset_index(name='txs')
        
        # Per-wallet aggregation
        block_metrics = {}
        
        for wallet in merged['wallet'].unique():
            wallet_data = merged[merged['wallet'] == wallet]
            wallet_txs = txs_per_block[txs_per_block['recipient'] == wallet]
            
            # Block capture metrics
            dominant_blocks = (wallet_data['block_share'] > 0.5).sum()
            total_blocks = len(wallet_data)
            
            block_metrics[wallet] = {
                'block_capture_rate': round(dominant_blocks / total_blocks, 4) if total_blocks > 0 else 0.0,
                'avg_block_share': round(wallet_data['block_share'].mean(), 4),
                'median_block_share': round(wallet_data['block_share'].median(), 4),
                'max_block_share': round(wallet_data['block_share'].max(), 4),
                'avg_txs_per_block': round(wallet_txs['txs'].mean(), 2),
                'multi_tx_rate': round((wallet_txs['txs'] > 1).mean(), 4),
                'n_blocks': total_blocks
            }
        
        print(f"   Calculated block metrics for {len(block_metrics):,} wallets")
        
        return block_metrics
