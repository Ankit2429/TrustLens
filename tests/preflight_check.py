"""Live Pre-flight Credential and Connectivity Verification.
Ensures zero secrets or private keys are ever printed.
"""
import os
import requests
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

serpapi_key = os.environ.get("SERPAPI_KEY")
pinata_jwt = os.environ.get("PINATA_JWT")
priv_key = os.environ.get("PRIVATE_KEY")
rpc_url = os.environ.get("AMOY_RPC_URL", "https://polygon-amoy.drpc.org")

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

# 3. Polygon Amoy RPC validation
try:
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 15}))
    if w3.is_connected():
        chain_id = w3.eth.chain_id
        if chain_id == 80002:
            results["POLYGON AMOY RPC"] = (True, f"Connected (Chain ID: {chain_id}, Block: {w3.eth.block_number})")
        else:
            results["POLYGON AMOY RPC"] = (False, f"Unexpected Chain ID: {chain_id} (expected 80002)")
    else:
        results["POLYGON AMOY RPC"] = (False, "Could not connect to RPC endpoint")
except Exception as e:
    results["POLYGON AMOY RPC"] = (False, f"RPC error: {type(e).__name__}")

# 4. Wallet & Balance validation
try:
    if not priv_key or priv_key == "your_testnet_wallet_private_key":
        results["WALLET"] = (False, "PRIVATE_KEY not configured or placeholder")
        results["TEST POL BALANCE"] = (False, "No wallet available")
    else:
        clean_key = priv_key.strip()
        if clean_key.startswith("0x"):
            clean_key = clean_key[2:]
        if len(clean_key) != 64 or not all(c in "0123456789abcdefABCDEF" for c in clean_key):
            results["WALLET"] = (
                False,
                f"Invalid private key format. Expected 64 hexadecimal characters (32 bytes). Found {len(clean_key)} non-hex characters."
            )
            results["TEST POL BALANCE"] = (False, "Cannot derive wallet address from invalid private key format")
        else:
            acct = w3.eth.account.from_key("0x" + clean_key)
            addr = acct.address
            results["WALLET"] = (True, f"Derived address: {addr}")

            if results["POLYGON AMOY RPC"][0]:
                bal_wei = w3.eth.get_balance(addr)
                bal_eth = float(w3.from_wei(bal_wei, "ether"))
                if bal_eth >= 0.005:
                    results["TEST POL BALANCE"] = (True, f"{bal_eth:.4f} POL (Sufficient for deployment and transactions)")
                elif bal_eth > 0:
                    results["TEST POL BALANCE"] = (True, f"{bal_eth:.6f} POL (Non-zero, may cover minimal transactions)")
                else:
                    results["TEST POL BALANCE"] = (False, f"0.0 POL (Insufficient funds. Please fund {addr} via https://faucet.polygon.technology/)")
            else:
                results["TEST POL BALANCE"] = (False, "RPC unavailable for balance check")
except Exception as e:
    results["WALLET"] = (False, f"Wallet error: {type(e).__name__}")
    results["TEST POL BALANCE"] = (False, f"Balance error: {type(e).__name__}")

print("==================================================")
print("  PHASE 3 LIVE PRE-FLIGHT VALIDATION")
print("==================================================")
for k, (ok, msg) in results.items():
    status_str = "READY" if (k != "TEST POL BALANCE" and ok) else ("SUFFICIENT" if (k == "TEST POL BALANCE" and ok) else ("INSUFFICIENT" if k == "TEST POL BALANCE" else "FAILED"))
    print(f"{k}: {status_str}")
    print(f"  -> {msg}")
print("==================================================")
