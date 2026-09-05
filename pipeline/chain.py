"""Network-agnostic on-chain read/write adapter for ProofRegistry smart contract.

Supports:
1. Foundry Anvil local EVM node (Chain ID 31337, http://127.0.0.1:8545)
2. Polygon Amoy public testnet (Chain ID 80002)
3. Dynamic configuration through unified BLOCKCHAIN_* environment variables
4. Fallback resilience, clean nonce/gas handling, and tamper-proof verification
"""
import json
import os
from pathlib import Path
from typing import Any, Tuple

from web3 import Web3
from web3.exceptions import ContractLogicError

# Default Anvil Account #0 Private Key (Publicly known local dev key for Foundry Anvil)
DEFAULT_ANVIL_PRIVATE_KEY = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)

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


def get_blockchain_config() -> dict[str, Any]:
    """Resolve active blockchain network configuration from environment variables.

    Supported networks:
        - 'anvil' / 'localhost' / 'local' (default: Chain ID 31337, http://127.0.0.1:8545)
        - 'polygon_amoy' / 'amoy' (Chain ID 80002, https://polygon-amoy.drpc.org)
    """
    raw_net = os.environ.get("BLOCKCHAIN_NETWORK", "anvil").strip().lower()
    is_anvil = raw_net in ("anvil", "localhost", "local", "hardhat")

    if is_anvil:
        rpc_url = os.environ.get("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
        chain_id = int(os.environ.get("BLOCKCHAIN_CHAIN_ID", "31337"))
        priv_key = (
            os.environ.get("BLOCKCHAIN_PRIVATE_KEY")
            or DEFAULT_ANVIL_PRIVATE_KEY
        )
        network_label = "Anvil Local Node"
    else:
        rpc_url = (
            os.environ.get("BLOCKCHAIN_RPC_URL")
            or os.environ.get("AMOY_RPC_URL")
            or "https://polygon-amoy.drpc.org"
        )
        chain_id = int(os.environ.get("BLOCKCHAIN_CHAIN_ID", "80002"))
        priv_key = os.environ.get("BLOCKCHAIN_PRIVATE_KEY") or os.environ.get("PRIVATE_KEY")
        network_label = "Polygon Amoy"

    contract_addr = (
        os.environ.get("BLOCKCHAIN_CONTRACT_ADDRESS")
        or os.environ.get("CONTRACT_ADDRESS")
    )

    return {
        "network": "anvil" if is_anvil else "polygon_amoy",
        "network_label": network_label,
        "is_local": is_anvil,
        "rpc_url": rpc_url,
        "chain_id": chain_id,
        "private_key": priv_key,
        "contract_address": contract_addr,
    }


def get_web3_and_contract() -> Tuple[Web3, Any]:
    """Connect to configured RPC and instantiate the ProofRegistry contract instance.

    Raises:
        EnvironmentError: If CONTRACT_ADDRESS / BLOCKCHAIN_CONTRACT_ADDRESS is missing.
        ConnectionError: If no RPC endpoint is reachable.
    """
    config = get_blockchain_config()
    contract_addr = config.get("contract_address")

    if not contract_addr or contract_addr in (
        "fill_in_after_running_deploy_script",
        "0x0000000000000000000000000000000000000000",
    ):
        deploy_cmd = "npm run deploy:anvil" if config["is_local"] else "npm run deploy:amoy"
        raise EnvironmentError(
            f"Contract address is not configured for {config['network_label']}. "
            f"Please deploy the contract (`{deploy_cmd}`) and set BLOCKCHAIN_CONTRACT_ADDRESS in .env."
        )

    w3 = None
    if config["is_local"]:
        # Direct connection to local node
        try:
            local_w3 = Web3(Web3.HTTPProvider(config["rpc_url"], request_kwargs={"timeout": 5}))
            if local_w3.is_connected():
                w3 = local_w3
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to local Anvil node at {config['rpc_url']}. "
                "Ensure Anvil is running with `anvil` or `anvil --port 8545`. "
                f"Underlying error: {e}"
            ) from e

        if w3 is None or not w3.is_connected():
            raise ConnectionError(
                f"Local Anvil node at {config['rpc_url']} is unreachable. "
                "Run `anvil` in a separate terminal."
            )
    else:
        # Public network with fallback candidates
        rpc_candidates = [
            config["rpc_url"],
            "https://polygon-amoy-bor-rpc.publicnode.com",
            "https://polygon-amoy.drpc.org",
            "https://rpc.ankr.com/polygon_amoy",
        ]
        seen_rpcs = set()
        unique_rpcs = []
        for r in rpc_candidates:
            if r and r not in seen_rpcs:
                seen_rpcs.add(r)
                unique_rpcs.append(r)

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
            raise ConnectionError(
                f"Could not connect to any {config['network_label']} RPC endpoints. Last error: {last_err}"
            )

    contract_checksum = Web3.to_checksum_address(contract_addr)
    contract = w3.eth.contract(address=contract_checksum, abi=load_abi())
    return w3, contract


def _calculate_amoy_gas_fees(w3: Web3) -> Tuple[int, int]:
    """Calculate dynamic EIP-1559 gas fees suitable for Polygon Amoy testnet.

    Returns:
        (maxPriorityFeePerGas, maxFeePerGas) in Wei.
    """
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
    """Register an evidence proof on the active blockchain network.

    Args:
        data_hash_hex: 64-character SHA-256 hex string of the canonical record.
        ipfs_cid: IPFS CID of the pinned evidence manifest.

    Returns:
        Dictionary containing transaction hash, block number, status, and network info.

    Raises:
        EnvironmentError: If private key is missing or invalid.
        RuntimeError: If transaction fails, reverts (duplicate proof), or lacks gas.
    """
    config = get_blockchain_config()
    priv_key = config.get("private_key")

    if not priv_key or priv_key in (
        "your_testnet_wallet_private_key",
        "your_anvil_private_key",
    ):
        raise EnvironmentError(
            f"Missing or unconfigured private key for {config['network_label']}. "
            "Please configure BLOCKCHAIN_PRIVATE_KEY in .env."
        )

    w3, contract = get_web3_and_contract()
    acct = w3.eth.account.from_key(priv_key)
    data_hash_bytes = _clean_hash_bytes(data_hash_hex)
    nonce = w3.eth.get_transaction_count(acct.address, "pending")

    # Estimate gas limit
    try:
        gas_est = contract.functions.registerProof(data_hash_bytes, ipfs_cid).estimate_gas(
            {"from": acct.address}
        )
        gas_limit = int(gas_est * 1.25)
    except ContractLogicError as e:
        if "Already registered" in str(e):
            raise RuntimeError(
                f"Proof already registered on-chain for hash 0x{data_hash_hex}. "
                f"{config['network_label']} ProofRegistry prevents duplicate overwrites."
            ) from e
        gas_limit = 250_000
    except Exception:
        gas_limit = 250_000

    # Build network-appropriate transaction
    if config["is_local"]:
        # Simple reliable gas for local Anvil
        gas_price = w3.eth.gas_price
        tx = contract.functions.registerProof(data_hash_bytes, ipfs_cid).build_transaction({
            "from": acct.address,
            "nonce": nonce,
            "chainId": config["chain_id"],
            "gas": gas_limit,
            "gasPrice": gas_price,
        })
        receipt_timeout = 15
    else:
        # Dynamic EIP-1559 for Polygon Amoy
        priority_fee, max_fee = _calculate_amoy_gas_fees(w3)
        tx = contract.functions.registerProof(data_hash_bytes, ipfs_cid).build_transaction({
            "from": acct.address,
            "nonce": nonce,
            "chainId": config["chain_id"],
            "gas": gas_limit,
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority_fee,
        })
        receipt_timeout = 120

    signed = acct.sign_transaction(tx)
    try:
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    except Exception as e:
        err_str = str(e)
        if "insufficient funds" in err_str.lower():
            faucet_hint = (
                "Ensure local Anvil account has ETH"
                if config["is_local"]
                else "Get free Amoy testnet tokens from https://faucet.polygon.technology/"
            )
            raise RuntimeError(
                f"Insufficient balance in wallet {acct.address} on {config['network_label']}. {faucet_hint}"
            ) from e
        raise RuntimeError(f"Failed to submit transaction to {config['network_label']}: {e}") from e

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=receipt_timeout)
    if receipt.status != 1:
        raise RuntimeError(f"Transaction reverted on-chain (tx: {tx_hash.hex()})")

    return {
        "tx_hash": tx_hash.hex(),
        "block": receipt.blockNumber,
        "status": "SUCCESS" if receipt.status == 1 else "FAILED",
        "submitter": acct.address,
        "network": config["network_label"],
        "chain_id": config["chain_id"],
        "contract_address": contract.address,
    }


def get_proof(data_hash_hex: str) -> dict[str, Any]:
    """Look up an evidence proof by its SHA-256 hash on the active blockchain.

    Args:
        data_hash_hex: 64-character SHA-256 hex string.

    Returns:
        Dictionary containing submitter address, ipfs_cid, timestamp, and network metadata.

    Raises:
        ValueError: If proof is not found on-chain.
    """
    config = get_blockchain_config()
    _, contract = get_web3_and_contract()
    data_hash_bytes = _clean_hash_bytes(data_hash_hex)

    try:
        submitter, ipfs_cid, timestamp = contract.functions.getProof(data_hash_bytes).call()
    except ContractLogicError as e:
        raise ValueError(
            f"Proof with hash 0x{data_hash_hex} not found on {config['network_label']}."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to query ProofRegistry contract on {config['network_label']}: {e}"
        ) from e

    if timestamp == 0 or not ipfs_cid:
        raise ValueError(
            f"Proof with hash 0x{data_hash_hex} is not registered on {config['network_label']}."
        )

    return {
        "data_hash": data_hash_hex,
        "submitter": submitter,
        "ipfs_cid": ipfs_cid,
        "timestamp": timestamp,
        "network": config["network_label"],
        "chain_id": config["chain_id"],
        "contract_address": contract.address,
    }
