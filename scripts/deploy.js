const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Deploying ProofRegistry to Polygon Amoy (Chain ID 80002)...");
  const [deployer] = await hre.ethers.getSigners();
  if (!deployer) {
    throw new Error("No deployer account configured. Check PRIVATE_KEY in .env.");
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

  // Automatically update CONTRACT_ADDRESS in .env safely
  const envPath = path.resolve(__dirname, "..", ".env");
  if (fs.existsSync(envPath)) {
    let envContent = fs.readFileSync(envPath, "utf8");
    if (envContent.includes("CONTRACT_ADDRESS=")) {
      envContent = envContent.replace(/CONTRACT_ADDRESS=.*/g, `CONTRACT_ADDRESS=${address}`);
    } else {
      envContent += `\nCONTRACT_ADDRESS=${address}\n`;
    }
    fs.writeFileSync(envPath, envContent, "utf8");
    console.log("Updated CONTRACT_ADDRESS in .env");
  }
}

main().catch((error) => {
  console.error("Deployment failed:", error.message || error);
  process.exitCode = 1;
});
