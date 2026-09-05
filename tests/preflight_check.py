"""Live Pre-flight Credential and Connectivity Verification.
Supports both Anvil Local Node and Polygon Amoy testnet.
Ensures zero secrets or private keys are ever printed.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from dotenv import load_dotenv
from web3 import Web3

from pipeline.chain import get_blockchain_config

load_dotenv()

serpapi_key = os.environ.get("SERPAPI_KEY")
pinata_jwt = os.environ.get("PINATA_JWT")

config = get_blockchain_config()
priv_key = config.get("private_key")
rpc_url = config.get("rpc_url")
network_label = config.get("network_label", "Blockchain")
expected_chain_id = config.get("chain_id")

results = {}

# 1. SerpApi validation
try:
    if not serpapi_key or serpapi_key == "your_serpapi_key":
        results["SERPAPI"] = (False, "SERPAPI_KEY not configured or placeholder")
    else:
        r = requests.get("https://serpapi.com/account", params={"api_key": serpapi_key}, timeout=15)
        if r.status_code == 200:
            results["SERPAPI"] = (True, "Authentication successful")
        else:
            try:
                err_msg = r.json().get("error", f"HTTP {r.status_code}")
            except Exception:
                err_msg = f"HTTP {r.status_code}"
            results["SERPAPI"] = (False, f"Authentication error: {err_msg}")
except Exception as e:
    results["SERPAPI"] = (False, f"Request failed: {type(e).__name__}")

# 2. Pinata validation
try:
    if not pinata_jwt or pinata_jwt == "your_pinata_jwt":
        results["PINATA"] = (False, "PINATA_JWT not configured or placeholder")
    else:
        r = requests.get(
            "https://api.pinata.cloud/data/testAuthentication",
            headers={"Authorization": f"Bearer {pinata_jwt}"},
            timeout=15,
        )
        if r.status_code == 200:
            results["PINATA"] = (True, "Authentication successful")
        else:
            results["PINATA"] = (False, f"Authentication error: HTTP {r.status_code}")
except Exception as e:
    results["PINATA"] = (False, f"Request failed: {type(e).__name__}")

# 3. Blockchain RPC validation
rpc_key = f"{network_label.upper()} RPC"
balance_key = f"{network_label.upper()} BALANCE"

try:
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 8}))
    if w3.is_connected():
        chain_id = w3.eth.chain_id
        if chain_id == expected_chain_id:
            results[rpc_key] = (True, f"Connected to {rpc_url} (Chain ID: {chain_id}, Block: {w3.eth.block_number})")
        else:
            results[rpc_key] = (False, f"Connected, but unexpected Chain ID: {chain_id} (expected {expected_chain_id})")
    else:
        results[rpc_key] = (False, f"Could not connect to RPC endpoint {rpc_url}")
except Exception as e:
    results[rpc_key] = (False, f"RPC error: {type(e).__name__} ({e})")

# 4. Wallet & Balance validation
try:
    if not priv_key or priv_key in ("your_testnet_wallet_private_key", "your_anvil_private_key"):
        results["WALLET"] = (False, "BLOCKCHAIN_PRIVATE_KEY / PRIVATE_KEY not configured or placeholder")
        results[balance_key] = (False, "No wallet available")
    else:
        clean_key = priv_key.strip()
        if clean_key.startswith("0x"):
            clean_key = clean_key[2:]
        if len(clean_key) != 64 or not all(c in "0123456789abcdefABCDEF" for c in clean_key):
            results["WALLET"] = (
                False,
                f"Invalid private key format. Expected 64 hex characters. Found {len(clean_key)}."
            )
            results[balance_key] = (False, "Cannot derive wallet address from invalid private key format")
        else:
            acct = w3.eth.account.from_key("0x" + clean_key)
            addr = acct.address
            results["WALLET"] = (True, f"Derived address: {addr}")

            if results.get(rpc_key, (False,))[0]:
                bal_wei = w3.eth.get_balance(addr)
                bal_eth = float(w3.from_wei(bal_wei, "ether"))
                currency = "ETH" if config["is_local"] else "POL"
                if bal_eth >= 0.005:
                    results[balance_key] = (True, f"{bal_eth:.4f} {currency} (Sufficient for transactions)")
                elif bal_eth > 0:
                    results[balance_key] = (True, f"{bal_eth:.6f} {currency} (Non-zero)")
                else:
                    results[balance_key] = (False, f"0.0 {currency} (Insufficient balance)")
            else:
                results[balance_key] = (False, "RPC unavailable for balance check")
except Exception as e:
    results["WALLET"] = (False, f"Wallet error: {type(e).__name__}")
    results[balance_key] = (False, f"Balance error: {type(e).__name__}")

print("==================================================")
print(f"  PRE-FLIGHT VALIDATION [{network_label.upper()}]")
print("==================================================")
for k, (ok, msg) in results.items():
    status_str = "READY" if ("BALANCE" not in k and ok) else ("SUFFICIENT" if ("BALANCE" in k and ok) else ("INSUFFICIENT" if "BALANCE" in k else "FAILED"))
    print(f"{k}: {status_str}")
    print(f"  -> {msg}")
print("==================================================")
