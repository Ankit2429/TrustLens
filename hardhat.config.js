require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

const { AMOY_RPC_URL, PRIVATE_KEY } = process.env;

const rawKey = process.env.PRIVATE_KEY ? process.env.PRIVATE_KEY.trim() : "";
const validKey =
  rawKey.length === 64 || (rawKey.startsWith("0x") && rawKey.length === 66)
    ? [rawKey.startsWith("0x") ? rawKey : "0x" + rawKey]
    : [];

module.exports = {
  solidity: "0.8.20",
  networks: {
    amoy: {
      url: AMOY_RPC_URL || "https://polygon-amoy.drpc.org",
      accounts: validKey,
      chainId: 80002,
    },
  },
};
