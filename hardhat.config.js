require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

const { AMOY_RPC_URL, PRIVATE_KEY } = process.env;

const anvilRawKey = process.env.BLOCKCHAIN_PRIVATE_KEY?.trim() || "";
const anvilAccounts =
  anvilRawKey.length === 64 || (anvilRawKey.startsWith("0x") && anvilRawKey.length === 66)
    ? [anvilRawKey.startsWith("0x") ? anvilRawKey : "0x" + anvilRawKey]
    : [];

const amoyRawKey =
  process.env.BLOCKCHAIN_PRIVATE_KEY?.trim() ||
  process.env.PRIVATE_KEY?.trim() ||
  "";
const amoyAccounts =
  amoyRawKey.length === 64 || (amoyRawKey.startsWith("0x") && amoyRawKey.length === 66)
    ? [amoyRawKey.startsWith("0x") ? amoyRawKey : "0x" + amoyRawKey]
    : [];

module.exports = {
  solidity: "0.8.20",
  networks: {
    localhost: {
      url: process.env.BLOCKCHAIN_RPC_URL || "http://127.0.0.1:8545",
      chainId: 31337,
      ...(anvilAccounts.length > 0 ? { accounts: anvilAccounts } : {}),
    },
    anvil: {
      url: process.env.BLOCKCHAIN_RPC_URL || "http://127.0.0.1:8545",
      chainId: 31337,
      ...(anvilAccounts.length > 0 ? { accounts: anvilAccounts } : {}),
    },
    amoy: {
      url: process.env.AMOY_RPC_URL || process.env.BLOCKCHAIN_RPC_URL || "https://polygon-amoy.drpc.org",
      accounts: amoyAccounts,
      chainId: 80002,
    },
  },
};
