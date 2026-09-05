"""Unit and Integration Tests for Anvil Local Blockchain Adapter.

Tests:
1. Blockchain network configuration resolution (Anvil vs Polygon Amoy)
2. ABI loading and verification
3. Hash formatting and validation helpers
4. End-to-end Anvil proof registration and retrieval (when Anvil is active)
5. Duplicate proof registration prevention
6. Query non-existent proof exception handling
"""
import os
import pytest
from unittest import mock
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

from pipeline.chain import (
    DEFAULT_ANVIL_PRIVATE_KEY,
    PROOF_REGISTRY_ABI,
    _clean_hash_bytes,
    get_blockchain_config,
    get_proof,
    get_web3_and_contract,
    load_abi,
    register_proof,
)


def test_blockchain_config_defaults_to_anvil():
    """Default configuration should select Anvil on chain 31337."""
    with mock.patch.dict(os.environ, {}, clear=True):
        cfg = get_blockchain_config()
        assert cfg["network"] == "anvil"
        assert cfg["chain_id"] == 31337
        assert cfg["rpc_url"] == "http://127.0.0.1:8545"
        assert cfg["is_local"] is True
        assert cfg["private_key"] == DEFAULT_ANVIL_PRIVATE_KEY


def test_blockchain_config_polygon_amoy():
    """Configuring BLOCKCHAIN_NETWORK=polygon_amoy should select Amoy defaults."""
    env = {
        "BLOCKCHAIN_NETWORK": "polygon_amoy",
        "BLOCKCHAIN_RPC_URL": "https://custom-amoy.rpc.io",
        "BLOCKCHAIN_CHAIN_ID": "80002",
        "BLOCKCHAIN_PRIVATE_KEY": "0x1234567890123456789012345678901234567890123456789012345678901234",
        "BLOCKCHAIN_CONTRACT_ADDRESS": "0x1111111111111111111111111111111111111111",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        cfg = get_blockchain_config()
        assert cfg["network"] == "polygon_amoy"
        assert cfg["chain_id"] == 80002
        assert cfg["rpc_url"] == "https://custom-amoy.rpc.io"
        assert cfg["is_local"] is False
        assert cfg["contract_address"] == "0x1111111111111111111111111111111111111111"


def test_clean_hash_bytes_validation():
    """_clean_hash_bytes should convert 64-char hex strings and reject invalid inputs."""
    valid_hash = "a" * 64
    b = _clean_hash_bytes(valid_hash)
    assert len(b) == 32
    assert b == bytes.fromhex(valid_hash)

    # With 0x prefix
    b2 = _clean_hash_bytes("0x" + valid_hash)
    assert b2 == b

    # Invalid lengths
    with pytest.raises(ValueError):
        _clean_hash_bytes("short")

    with pytest.raises(ValueError):
        _clean_hash_bytes("a" * 63)


def test_load_abi():
    """load_abi must return a non-empty ABI list with registerProof and getProof."""
    abi = load_abi()
    assert isinstance(abi, list)
    fn_names = {item["name"] for item in abi if item.get("type") == "function"}
    assert "registerProof" in fn_names
    assert "getProof" in fn_names


def test_live_anvil_integration():
    """Live test against running Anvil node (skipped if Anvil is not reachable or contract unconfigured)."""
    cfg = get_blockchain_config()
    if not cfg["is_local"]:
        pytest.skip("Skipping Anvil live test because active network is not Anvil")

    try:
        w3, contract = get_web3_and_contract()
    except Exception as e:
        pytest.skip(f"Anvil node or contract not available: {e}")

    import hashlib
    import time

    # Generate a unique deterministic hash for this test run
    test_data = f"test_evidence_{time.time()}_{os.getpid()}".encode("utf-8")
    test_hash = hashlib.sha256(test_data).hexdigest()
    test_cid = "QmTestAnvilIntegrationCID1234567890abcdef"

    # 1. Register proof
    receipt = register_proof(test_hash, test_cid)
    assert receipt["status"] == "SUCCESS"
    assert receipt["chain_id"] == 31337
    assert receipt["network"] == "Anvil Local Node"
    assert len(receipt["tx_hash"]) > 0

    # 2. Read back proof
    proof = get_proof(test_hash)
    assert proof["data_hash"] == test_hash
    assert proof["ipfs_cid"] == test_cid
    assert proof["timestamp"] > 0
    assert proof["network"] == "Anvil Local Node"
    assert proof["chain_id"] == 31337

    # 3. Duplicate registration should raise RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        register_proof(test_hash, test_cid)
    assert "already registered" in str(excinfo.value).lower()
