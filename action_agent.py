import argparse
import logging
import os
import pprint
from logging import getLogger
from datetime import date, timedelta

import numpy as np
import torch
import yfinance as yf
from dotenv import find_dotenv, load_dotenv
from giza.agents.model import GizaModel
from sklearn.preprocessing import MinMaxScaler

from addresses import ADDRESSES
from starknet_py.net.models import StarknetChainId
from starknet_py.net.account.account import Account
from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.net.signer.stark_curve_signer import KeyPair
from avnu_helpers import approve, swap_eth_to_usdc, get_balance, check_allowance, swap_usdc_to_eth

load_dotenv(find_dotenv())

dev_private_key = os.environ.get("PRIVATE_KEY")
dev_public_key = os.environ.get("PUBLIC_KEY")
starknet_rpc_url = os.environ.get("STARKNET_RPC_URL")
dry_run = os.environ.get("DRY_RUN")

logging.basicConfig(level=logging.INFO)

# Fetch Historical Price Data
def fetch_data(ticker, start_date):
    data = yf.download(ticker, start=start_date)
    return data['Close'].values.reshape(-1, 1)

# Preprocess Data
def preprocess_data(data, window_size):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    last_window = data[-window_size:]
    last_window_scaled = scaler.transform(last_window)
    last_window_scaled = torch.tensor(last_window_scaled, dtype=torch.float32).view(1, -1)   

    last_window_scaled = np.array(last_window_scaled).reshape(1, -1)
    return last_window_scaled, scaler

def create_model_and_account(
    model_id: int, version_id: int
):
    """
    Create a Giza model and a Starknet account.
    """

    # we create an instance of our Account
    node_url = "https://free-rpc.nethermind.io/mainnet-juno"
    client = FullNodeClient(node_url=node_url)

    account = Account(
        address=dev_public_key,
        client=client,
        key_pair=KeyPair.from_private_key(dev_private_key),
        chain=StarknetChainId.MAINNET,
    )

    model = GizaModel(
        id=model_id,
        version=version_id
    )
    return model, account

def predict(model: GizaModel, last_window_scaled: np.ndarray, scaler: MinMaxScaler):
    """
    Predict the ETH price.

    Args:
        X (np.ndarray): Input to the model.

    Returns:
        float: Predicted ETH price.
        string: Request ID.  
    """
    (result, request_id) = model.predict(
        input_feed={"last_window_scaled" :last_window_scaled}, verifiable=True, dry_run=True
    )

    # convert result to tensor  
    tensor_result = torch.tensor(result)
    predicted_price = scaler.inverse_transform(tensor_result.numpy().reshape(-1, 1))
    eth_price_prediction = predicted_price[0,0]

    return eth_price_prediction, request_id

def execute_trading_strategy(model_id: int, version_id: int):
    """
    Execute the trading strategy based on the prediction.
    """
    logger = getLogger("agent_logger")
    nft_manager_address = ADDRESSES["NonfungiblePositionManager"][11155111]
    
    logger.info("Starting the trading strategy")

    # Parameters
    ticker = "ETH-USD"
    start_date = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")
    window_size = 60
    
    # Fetch and preprocess data
    logger.info("Fetching last 60 trading sessions data")
    data = fetch_data(ticker, start_date)
    (last_window_scaled,scaler) = preprocess_data(data, window_size)

    # Create the model and account
    logger.info("Creating the model and account")
    (model, account) = create_model_and_account(model_id, version_id)

    # Predict the price
    logger.info("Predicting the price")
    (eth_price_prediction, request_id) = predict(model, last_window_scaled, scaler)
    logger.info(f"Predicted ETH price: {eth_price_prediction}")

    # Execute the trading logic
    # get current ETH price from Yahoo Finance
    current_eth_price = yf.Ticker("ETH-USD").history(period="1d")["Close"].values[0]
    logger.info(f"Current ETH price: {current_eth_price}")

    # if the predicted price is more than 1% higher than the current price, buy
    if eth_price_prediction > current_eth_price * 1.01:
        logger.info("Buying ETH")
        # get usdc balance , if > 0.2 , buy ETH with 0.1 USDC
        # account.buy_eth(eth_price_prediction, 0.1)
    elif eth_price_prediction < current_eth_price * 0.99:
        logger.info("Selling ETH")
        # get ETH balance ,sell all ETH for USDC
        # account.sell_eth(eth_price_prediction, 0.1)
    else:
        logger.info("No trading action needed at this time.")
   
    logger.info("Finished")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model-id", metavar="M", type=int, help="model-id")
    parser.add_argument("--version-id", metavar="V", type=int, help="version-id")
    
    # Parse arguments
    args = parser.parse_args()

    MODEL_ID = args.model_id
    VERSION_ID = args.version_id

    execute_trading_strategy( MODEL_ID, VERSION_ID)