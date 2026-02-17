# alchemy_lib.py MEV bot profiler library

import asyncio
import aiohttp
from aiolimiter import AsyncLimiter

class RPCProfiler:
    def __init__(self, rpc_url):
        self.rpc_url = rpc_url
        self.limiter = AsyncLimiter(10, 1) 

    async def get_rpc_code(self, session, wallet):
        clean_wallet = wallet.lower().strip()
        payloads = [
            {"jsonrpc": "2.0", "id": 1, "method": "eth_getCode", "params": [clean_wallet, "latest"]},
            {"jsonrpc": "2.0", "id": 2, "method": "eth_getBalance", "params": [clean_wallet, "latest"]}
        ]
        
        async with self.limiter:
            try:
                async with session.post(self.rpc_url, json=payloads, timeout=10) as resp:
                    data = await resp.json()
                    
                    # Robust extraction: check for 'result' in both batch items
                    code_res = data[0].get("result") if isinstance(data, list) and len(data) > 0 else "0x"
                    bal_hex = data[1].get("result") if isinstance(data, list) and len(data) > 1 else "0x0"
                    
                    # Fallback if result is None (sometimes happens on certain node errors)
                    code_res = code_res or "0x"
                    bal_hex = bal_hex or "0x0"

                    bal_eth = int(bal_hex, 16) / 10**18
                    actual_bytecode = code_res[2:] 
                    b_len = len(actual_bytecode) // 2 

                    return {
                        "wallet": clean_wallet,
                        "bytecode_len": b_len,
                        "eth_balance": bal_eth,
                        "is_contract": b_len > 0, 
                        "is_active_mev_proxy": b_len > 0 and bal_eth < 0.05
                    }
            except Exception as e:
                return {"wallet": clean_wallet, "bytecode_len": 0, "eth_balance": 0, "status": f"error: {str(e)}"}

async def fetch_all_profiles(wallet_list, rpc_url):
    if not rpc_url or "YOUR_" in rpc_url:
        print("⚠️ No valid RPC_URL found.")
        return {w.lower(): {"bytecode_len": 0} for w in wallet_list}

    connector = aiohttp.TCPConnector(limit=20) # Increased for speed
    profiler = RPCProfiler(rpc_url)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [profiler.get_rpc_code(session, w) for w in wallet_list]
        results = await asyncio.gather(*tasks)
        return {res['wallet']: res for res in results}
