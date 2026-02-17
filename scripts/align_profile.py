import csv
import asyncio
import os
from alchemy_lib import fetch_all_profiles  # Ensure alchemy_lib.py is in the same folder

# ================= CONFIG =================
RESULT_FILE = "result.csv"
LABEL_FILE = "mev_list.csv"
OUTPUT_FILE = "aligned_wallets_profiled.csv"

RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/xxx"

# ================= CORE LOGIC =================

async def main():
    # 1. Load Labels
    wallet_to_label = {}
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wallet = row["wallet"].strip().lower()
                wallet_to_label[wallet] = row.get("name_label", "mev_bot")
    
    # 2. Collect all unique wallets from Result File
    wallets_to_profile = []
    rows = []
    with open(RESULT_FILE, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        original_fieldnames = reader.fieldnames
        for row in reader:
            wallet = row["wallet"].strip().lower()
            wallets_to_profile.append(wallet)
            rows.append(row)

    print(f"🔍 Profiling {len(wallets_to_profile)} wallets via RPC...")

    # 3. Fetch RPC Profiles (Bytecode, Balance, is_contract)
    # This uses your alchemy_lib batching logic
    profiles = await fetch_all_profiles(wallets_to_profile, RPC_URL)

    # 4. Align and Write
    matched = 0
    new_fields = ["name_label", "bytecode_len", "eth_balance", "is_contract", "is_active_mev_proxy"]
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=original_fieldnames + new_fields)
        writer.writeheader()

        for row in rows:
            wallet = row["wallet"].strip().lower()
            
            # Get Label
            label = wallet_to_label.get(wallet, "unknown")
            if label != "unknown":
                matched += 1
            
            # Get RPC Data
            profile = profiles.get(wallet, {})
            
            # Merge data
            row["name_label"] = label
            row["bytecode_len"] = profile.get("bytecode_len", 0)
            row["eth_balance"] = profile.get("eth_balance", 0)
            row["is_contract"] = profile.get("is_contract", False)
            row["is_active_mev_proxy"] = profile.get("is_active_mev_proxy", False)
            
            writer.writerow(row)

    # 5. Final Report
    print("-" * 30)
    print(f"✅ Processed {len(rows)} total wallets.")
    print(f"🏷️  Matched {matched} from mev_list.csv.")
    print(f"🤖 Identified {sum(1 for p in profiles.values() if p.get('is_contract'))} Smart Contracts.")
    print(f"📄 Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
