"""On-chain read/write against the deployed ProofRegistry contract on Polygon Amoy (Chain ID 80002).

Features:
- Dynamic EIP-1559 gas pricing tailored for Polygon Amoy
- Embedded contract ABI fallback (resilient if Hardhat artifacts are not pre-built)
- Clear error handling for missing keys, duplicate registrations, and testnet gas shortages
"""
import json
import os
from pathlib import Path
from typing import Any, Tuple

from web3 import Web3
from web3.exceptions import ContractLogicError

# Fallback ABI in case artifacts are not built yet
PROOF_REGISTRY_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "ipfsCID", "type": "string"},
            {"indexed": True, "internalType": "address", "name": "submitter", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "name": "ProofRegistered",
        "type": "event",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "dataHash", "type": "bytes32"}],
        "name": "getProof",
        "outputs": [
            {"internalType": "address", "name": "submitter", "type": "address"},
            {"internalType": "string", "name": "ipfsCID", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
            {"internalType": "string", "name": "ipfsCID", "type": "string"},
        ],
        "name": "registerProof",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "name": "proofs",
        "outputs": [
            {"internalType": "address", "name": "submitter", "type": "address"},
            {"internalType": "string", "name": "ipfsCID", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts" / "contracts" / "ProofRegistry.sol" / "ProofRegistry.json"
)


def load_abi() -> list[dict[str, Any]]:
    """Load contract ABI from Hardhat artifact if available, otherwise return embedded fallback."""
    if ARTIFACT_PATH.exists():
        try:
            with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("abi", PROOF_REGISTRY_ABI)
        except Exception:
            pass
    return PROOF_REGISTRY_ABI


def get_web3_and_contract() -> Tuple[Web3, Any]:
    """Connect to Polygon Amoy RPC and instantiate the ProofRegistry contract instance.

    Uses automatic multi-RPC fallback for high availability.

    Raises:
        EnvironmentError: If CONTRACT_ADDRESS is missing.
        ConnectionError: If no RPC endpoint is reachable.
    """
    configured_rpc = os.environ.get("AMOY_RPC_URL", "https://polygon-amoy.drpc.org")
    contract_addr = os.environ.get("CONTRACT_ADDRESS")

    if not contract_addr or contract_addr == "fill_in_after_running_deploy_script":
        raise EnvironmentError(
            "CONTRACT_ADDRESS is not configured in .env. "
            "Please deploy the contract (`npm run deploy:amoy`) and set CONTRACT_ADDRESS."
        )

    rpc_candidates = [
        configured_rpc,
        "https://polygon-amoy-bor-rpc.publicnode.com",
        "https://polygon-amoy.drpc.org",
        "https://rpc.ankr.com/polygon_amoy",
    ]
    # Deduplicate while preserving order
    seen_rpcs = set()
    unique_rpcs = []
    for r in rpc_candidates:
        if r and r not in seen_rpcs:
            seen_rpcs.add(r)
            unique_rpcs.append(r)

    w3 = None
    last_err = None
    for rpc in unique_rpcs:
        try:
            candidate_w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 12}))
            if candidate_w3.is_connected():
                w3 = candidate_w3
                break
        except Exception as e:
            last_err = e

    if w3 is None:
        raise ConnectionError(f"Could not connect to any Polygon Amoy RPC endpoints. Last error: {last_err}")

    contract_checksum = Web3.to_checksum_address(contract_addr)
    contract = w3.eth.contract(address=contract_checksum, abi=load_abi())
    return w3, contract


def _calculate_amoy_gas_fees(w3: Web3) -> Tuple[int, int]:
    """Calculate dynamic EIP-1559 gas fees suitable for Polygon Amoy testnet.

    Returns:
        (maxPriorityFeePerGas, maxFeePerGas) in Wei.
    """
    # Amoy testnet requires a generous tip (minimum ~30-35 gwei)
    min_priority_fee = w3.to_wei(35, "gwei")
    try:
        current_priority = w3.eth.max_priority_fee
        priority_fee = max(current_priority, min_priority_fee)
    except Exception:
        priority_fee = min_priority_fee

    try:
        latest_block = w3.eth.get_block("latest")
        base_fee = latest_block.get("baseFeePerGas", w3.to_wei(30, "gwei"))
    except Exception:
        base_fee = w3.to_wei(30, "gwei")

    max_fee = (base_fee * 2) + priority_fee
    return priority_fee, max_fee


def _clean_hash_bytes(hash_hex: str) -> bytes:
    """Validate and convert 64-char hex string to 32 bytes."""
    clean = hash_hex.strip()
    if clean.startswith("0x"):
        clean = clean[2:]
    if len(clean) != 64:
        raise ValueError(f"Invalid SHA-256 hash length: expected 64 hex characters, got {len(clean)}")
    return bytes.fromhex(clean)


def register_proof(data_hash_hex: str, ipfs_cid: str) -> dict[str, Any]:
    """Register an evidence proof on Polygon Amoy.

    Args:
        data_hash_hex: 64-character SHA-256 hex string of the canonical record.
        ipfs_cid: IPFS CID of the pinned evidence manifest.

    Returns:
        Dictionary containing transaction hash, block number, status, and network info.

    Raises:
        EnvironmentError: If PRIVATE_KEY is missing or invalid.
        RuntimeError: If transaction fails, reverts (duplicate proof), or lacks gas.
    """
    priv_key = os.environ.get("PRIVATE_KEY")
    if not priv_key or priv_key == "your_testnet_wallet_private_key":
        raise EnvironmentError(
            "Missing or unconfigured PRIVATE_KEY in .env. "
            "Please configure your testnet wallet private key."
        )

    w3, contract = get_web3_and_contract()
    acct = w3.eth.account.from_key(priv_key)
    data_hash_bytes = _clean_hash_bytes(data_hash_hex)

    priority_fee, max_fee = _calculate_amoy_gas_fees(w3)
    nonce = w3.eth.get_transaction_count(acct.address, "pending")

    # Estimate gas if possible, fallback to 250k
    try:
        gas_est = contract.functions.registerProof(data_hash_bytes, ipfs_cid).estimate_gas(
            {"from": acct.address}
        )
        gas_limit = int(gas_est * 1.3)
    except ContractLogicError as e:
        if "Already registered" in str(e):
            raise RuntimeError(
                f"Proof already registered on-chain for hash 0x{data_hash_hex}. "
                "Polygon Amoy ProofRegistry prevents duplicate overwrites."
            ) from e
        gas_limit = 250_000
    except Exception:
        gas_limit = 250_000

    tx = contract.functions.registerProof(data_hash_bytes, ipfs_cid).build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "chainId": 80002,
        "gas": gas_limit,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority_fee,
    })

    signed = acct.sign_transaction(tx)
    try:
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    except Exception as e:
        err_str = str(e)
        if "insufficient funds" in err_str.lower():
            raise RuntimeError(
                f"Insufficient testnet POL in wallet {acct.address}. "
                "Get free Amoy testnet tokens from https://faucet.polygon.technology/"
            ) from e
        raise RuntimeError(f"Failed to submit transaction to Polygon Amoy: {e}") from e

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status != 1:
        raise RuntimeError(f"Transaction reverted on-chain (tx: {tx_hash.hex()})")

    return {
        "tx_hash": tx_hash.hex(),
        "block": receipt.blockNumber,
        "status": "SUCCESS" if receipt.status == 1 else "FAILED",
        "submitter": acct.address,
        "network": "Polygon Amoy",
        "chain_id": 80002,
        "contract_address": contract.address,
    }


def get_proof(data_hash_hex: str) -> dict[str, Any]:
    """Look up an evidence proof by its SHA-256 hash on Polygon Amoy.

    Args:
        data_hash_hex: 64-character SHA-256 hex string.

    Returns:
        Dictionary containing submitter address, ipfs_cid, and timestamp.

    Raises:
        ValueError: If proof is not found on-chain.
    """
    _, contract = get_web3_and_contract()
    data_hash_bytes = _clean_hash_bytes(data_hash_hex)

    try:
        submitter, ipfs_cid, timestamp = contract.functions.getProof(data_hash_bytes).call()
    except ContractLogicError as e:
        raise ValueError(f"Proof with hash 0x{data_hash_hex} not found on Polygon Amoy.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to query ProofRegistry contract: {e}") from e

    if timestamp == 0 or not ipfs_cid:
        raise ValueError(f"Proof with hash 0x{data_hash_hex} is not registered on-chain.")

    return {
        "data_hash": data_hash_hex,
        "submitter": submitter,
        "ipfs_cid": ipfs_cid,
        "timestamp": timestamp,
        "network": "Polygon Amoy",
        "chain_id": 80002,
        "contract_address": contract.address,
    }

