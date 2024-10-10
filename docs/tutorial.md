# Miner and Validator Tutorial

This tutorial will guide you through the process of setting up and running miners and validators for the AutoML network.

## Setup

### Prerequisite

If you don't have a key on the bittensor network refer to [this](https://docs.bittensor.com/getting-started/wallets). You can refer to [taostats.io](https://taostats.io) to find out where to buy TAO to cover your registration cost.

### 1. Install the library

```
git clone https://github.com/Hivetrain/DistributedAutoML
cd DistributedAutoML
sudo apt install git-lfs
git lfs install
pip install -r requirements.txt
pip install -e .
```

### 2. Register Your Miner/Validator

First, you need to register your miner on the network:

```
btcli s register
```

### 3. Configure Your Miner/Validator

Edit the config files in the configs/ folder to set the following configurations:

- Bittensor Config:
- `netuid`: Set the network UID (100 for testnet, 38 for mainnet)
- `wallet_name`: Set your wallet name
- `wallet_hotkey`: Set your wallet hotkey
- `network`: Set to "test" for testnet or "finney" for mainnet
- `subtensor_chain_endpoint`: Edit if using your own subtensor node
- General Config:
- `hf_token`: Set your Hugging Face token
- `gene_repo`: Set your Hugging Face repository name for storing genes (Miner only)
- Miner Config:
- Good luck ! Hyperparameter optimization here might help performance

### 4. Register Metadata (Miner only)

Run the following script to register your Hugging Face repository to the chain:

```
python register_miner.py
```

### 5. Run the Miner/Validator

Execute the miner script:

No autoupdate:
```
python neurons/miner.py
```
Autoupdate (Make sure pm2 is installed):
```
pm2 start pm2_miner.json
```

Execute the validator script:

No autoupdate:
```
python neurons/validator.py
```
Autoupdate (Make sure pm2 is [installed](https://pm2.io/docs/runtime/guide/installation/)):
```
pm2 start pm2_validator.json
```

## Additional Notes

- Ensure you have the required dependencies installed. You may need to run `pip install -r requirements.txt` (if a requirements file is provided).
- The `metrics_file` in `config.py` specifies where performance metrics will be saved.
- For both miners and validators, make sure you have sufficient balance in your wallet to pay for transaction fees.
- Monitor the console output and log files (`miner.log` for miners, `validator.log` for validators) for any errors or important information.
- The `device` setting in `config.py` determines whether to use CPU or GPU. Set it to "cuda" if you want to use a GPU.

Remember to keep your wallet information and Hugging Face token secure and never share them publicly.
