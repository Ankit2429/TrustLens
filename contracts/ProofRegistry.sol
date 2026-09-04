// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ProofRegistry
/// @notice Stores a tamper-evident hash of a discovered social-media/web post,
///         plus the IPFS CID of the full record, so anyone can re-verify it later.
contract ProofRegistry {
    struct Proof {
        address submitter;
        string ipfsCID;
        uint256 timestamp;
    }

    // dataHash (sha256 of the canonical JSON record) => Proof
    mapping(bytes32 => Proof) public proofs;

    event ProofRegistered(
        bytes32 indexed dataHash,
        string ipfsCID,
        address indexed submitter,
        uint256 timestamp
    );

    /// @notice Register a new proof. Reverts if this exact hash was already registered
    ///         (prevents silent overwrites — a resubmission needs a new hash).
    function registerProof(bytes32 dataHash, string calldata ipfsCID) external {
        require(proofs[dataHash].timestamp == 0, "Already registered");
        proofs[dataHash] = Proof({
            submitter: msg.sender,
            ipfsCID: ipfsCID,
            timestamp: block.timestamp
        });
        emit ProofRegistered(dataHash, ipfsCID, msg.sender, block.timestamp);
    }

    /// @notice Look up a proof by its data hash.
    function getProof(bytes32 dataHash)
        external
        view
        returns (address submitter, string memory ipfsCID, uint256 timestamp)
    {
        Proof memory p = proofs[dataHash];
        require(p.timestamp != 0, "Not found");
        return (p.submitter, p.ipfsCID, p.timestamp);
    }
}
