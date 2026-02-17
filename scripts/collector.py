# data collector

import time
import csv
import requests
import os
from datetime import datetime, timezone

# ================= CONFIG =================
RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/xxx"
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
DECIMALS_USDC = 6
DECIMALS_WETH = 18
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
OUTPUT_FILE = "forensic_dex_data.csv"


class ForensicCollectorV2:
    def __init__(self, rpc_url):
        self.rpc_url = rpc_url
        self.current_block = self._get_latest_block()
        self.last_price = None  # Track price across blocks

    # ---------------- RPC ----------------
    def _rpc_call(self, method, params):
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        try:
            r = requests.post(self.rpc_url, json=payload, timeout=15)
            return r.json().get("result")
        except Exception as e:
            print(f"⚠️ RPC error ({method}): {e}")
            return None

    def _get_latest_block(self):
        res = self._rpc_call("eth_blockNumber", [])
        return int(res, 16) if res else 0

    # ---------------- PRICE ----------------
    def get_pool_price_at_block(self, block_num):
        data = self._rpc_call(
            "eth_call",
            [{"to": POOL_ADDRESS, "data": "0x3850c7bd"}, hex(block_num)],
        )
        if not data:
            return None
        sqrtPriceX96 = int(data[2:66], 16)
        return self.decode_price(sqrtPriceX96)

    def decode_price(self, sqrtPriceX96):
        raw_price = (int(sqrtPriceX96) / 2 ** 96) ** 2
        return 1 / (raw_price * (10 ** DECIMALS_USDC / 10 ** DECIMALS_WETH))

    # ---------------- HELPERS ----------------
    def _to_signed_int(self, hex_str):
        val = int(hex_str, 16)
        return val if val < 2 ** 255 else val - 2 ** 256

    def _hex_to_int(self, val):
        try:
            return int(val, 16)
        except Exception:
            return None

    def _topic_to_address(self, topic):
        # last 20 bytes of 32-byte topic
        return "0x" + topic[-40:]

    # ---------------- CORE ----------------
    def fetch_trades(self, block_num):
        # Get baseline price for this block
        if self.last_price is None:
            # First run: fetch initial price
            self.last_price = self.get_pool_price_at_block(block_num - 1)
            if self.last_price is None:
                print(f"⚠️ Could not fetch initial price for block {block_num}")
                return 0

        current_running_price = self.last_price

        # Fetch block data
        block_data = self._rpc_call(
            "eth_getBlockByNumber", [hex(block_num), False]
        )

        if not block_data or "timestamp" not in block_data:
            print(f"⚠️ Skipping block {block_num}: no block data")
            return 0

        ts_int = self._hex_to_int(block_data.get("timestamp"))
        if ts_int is None:
            print(f"⚠️ Skipping block {block_num}: bad timestamp")
            return 0

        human_ts = datetime.fromtimestamp(
            ts_int, tz=timezone.utc
        ).isoformat()

        # Fetch swap logs
        logs = self._rpc_call(
            "eth_getLogs",
            [{
                "fromBlock": hex(block_num),
                "toBlock": hex(block_num),
                "address": POOL_ADDRESS,
                "topics": [SWAP_TOPIC],
            }],
        )

        if not logs:
            return 0

        sorted_logs = sorted(
            logs, key=lambda x: int(x.get("logIndex", "0x0"), 16)
        )

        # Cache transaction receipts to minimize RPC calls
        tx_cache = {}

        for log in sorted_logs:
            try:
                data = log["data"][2:]

                amt0 = self._to_signed_int(data[0:64]) / 10 ** DECIMALS_USDC
                amt1 = self._to_signed_int(data[64:128]) / 10 ** DECIMALS_WETH
                execution_price = self.decode_price(
                    int(data[128:192], 16)
                )

                side = "BUY_ETH" if amt0 > 0 else "SELL_ETH"
                impact_pct = (
                    (execution_price - current_running_price)
                    / current_running_price
                ) * 100

                tx_hash = log["transactionHash"]

                # -------- Fetch transaction data (cached) --------
                if tx_hash not in tx_cache:
                    receipt = self._rpc_call(
                        "eth_getTransactionReceipt", [tx_hash]
                    )
                    
                    if receipt:
                        gas_used = int(receipt.get("gasUsed", "0x0"), 16)
                        effective_gas_price = int(
                            receipt.get("effectiveGasPrice", "0x0"), 16
                        )
                        tx_cache[tx_hash] = {
                            "from": receipt.get("from", "Unknown"),
                            "gas_used": gas_used,
                            "gas_price_gwei": effective_gas_price / 1e9,
                            "tx_fee_eth": (gas_used * effective_gas_price) / 1e18
                        }
                    else:
                        tx_cache[tx_hash] = {
                            "from": "Unknown",
                            "gas_used": 0,
                            "gas_price_gwei": 0,
                            "tx_fee_eth": 0
                        }

                tx_data = tx_cache[tx_hash]

                # Extract recipient from indexed topics
                recipient = "Unknown"

                # Extract indexed addresses
                topics = log.get("topics", [])

                swap_sender = "Unknown"
                recipient = "Unknown"

                if len(topics) > 1:
                    swap_sender = self._topic_to_address(topics[1])

                if len(topics) > 2:
                    recipient = self._topic_to_address(topics[2])


                self.save_trade({
                    "block": block_num,
                    "log_index": int(log["logIndex"], 16),
                    "timestamp": human_ts,

                    # Role separation: sender vs recipient
                    "tx_sender": tx_data["from"],
                    "recipient": recipient,
                    "swap_sender": swap_sender,

                    "side": side,
                    "base_qty_eth": abs(amt1),
                    "quote_qty_usdc": abs(amt0),
                    "execution_price": round(execution_price, 4),
                    "impact_pct": round(impact_pct, 6),
                    "gas_used": tx_data["gas_used"],
                    "gas_price_gwei": round(tx_data["gas_price_gwei"], 2),
                    "tx_fee_eth": f"{tx_data['tx_fee_eth']:.8f}",
                    "tx_hash": tx_hash,
                })

                # Update running price for next swap in block
                current_running_price = execution_price

            except Exception as e:
                print(f"⚠️ Skipping log in block {block_num}: {e}")
                continue

        # Store last price for next block
        if len(sorted_logs) > 0:
            self.last_price = current_running_price

        return len(logs)

    # ---------------- STORAGE ----------------
    def save_trade(self, row):
        file_exists = os.path.isfile(OUTPUT_FILE)
        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ---------------- LOOP ----------------
    def run(self):
        print(f"🚀 Started tracking from block {self.current_block}")
        print(f"📊 Output file: {OUTPUT_FILE}")
        print(f"🏊 Pool: {POOL_ADDRESS}")
        print("-" * 60)
        
        while True:
            try:
                latest = self._get_latest_block()
                if self.current_block <= latest:
                    count = self.fetch_trades(self.current_block)
                    if count > 0:
                        print(f"✅ Block {self.current_block}: Found {count} trade(s)")
                    self.current_block += 1
                else:
                    # Caught up to latest block, wait for new blocks
                    time.sleep(2)
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user")
                break
            except Exception as e:
                print(f"❌ Error processing block {self.current_block}: {e}")
                time.sleep(5)  # Wait before retrying


if __name__ == "__main__":
    collector = ForensicCollectorV2(RPC_URL)
    collector.run()
