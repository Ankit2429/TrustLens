const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

function updateEnvFile(updates) {
  const envPath = path.resolve(__dirname, "..", ".env");
  if (!fs.existsSync(envPath)) return;

  let envContent = fs.readFileSync(envPath, "utf8");
  for (const [key, val] of Object.entries(updates)) {
    const regex = new RegExp(`^${key}=.*`, "m");
    if (regex.test(envContent)) {
      envContent = envContent.replace(regex, `${key}=${val}`);
    } else {
      envContent += `\n${key}=${val}`;
    }
  }
  fs.writeFileSync(envPath, envContent.trim() + "\n", "utf8");
}

async function main() {
  const network = await hre.ethers.provider.getNetwork();
  const chainId = Number(network.chainId);
  const isAnvil = chainId === 31337 || hre.network.name === "localhost" || hre.network.name === "anvil";
  const networkLabel = isAnvil ? "Anvil Local EVM Node (Chain ID 31337)" : `Polygon Amoy (Chain ID ${chainId})`;

  console.log(`Deploying ProofRegistry to ${networkLabel}...`);
  const [deployer] = await hre.ethers.getSigners();
  if (!deployer) {
    throw new Error(
      isAnvil
        ? "No signer available on local node. Ensure Anvil is running on port 8545."
        : "No deployer account configured. Check PRIVATE_KEY or BLOCKCHAIN_PRIVATE_KEY in .env."
    );
  }
  console.log("Deployer address:", deployer.address);

  const ProofRegistry = await hre.ethers.getContractFactory("ProofRegistry");
  const registry = await ProofRegistry.deploy();
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  const tx = registry.deploymentTransaction();
  const receipt = await tx.wait();

  console.log("ProofRegistry deployed successfully!");
  console.log("Contract address:", address);
  console.log("Transaction hash:", tx.hash);
  console.log("Block number:", receipt.blockNumber);

  const updates = {
    CONTRACT_ADDRESS: address,
    BLOCKCHAIN_CONTRACT_ADDRESS: address,
  };

  if (isAnvil) {
    updates.BLOCKCHAIN_NETWORK = "anvil";
    updates.BLOCKCHAIN_RPC_URL = "http://127.0.0.1:8545";
    updates.BLOCKCHAIN_CHAIN_ID = "31337";
    updates.BLOCKCHAIN_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
  } else {
    updates.BLOCKCHAIN_NETWORK = "polygon_amoy";
    updates.BLOCKCHAIN_CHAIN_ID = "80002";
  }

  updateEnvFile(updates);
  console.log("Updated .env with contract address and network configuration.");
}

main().catch((error) => {
  console.error("Deployment failed:", error.message || error);
  process.exitCode = 1;
});
