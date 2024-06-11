import argparse
import logging
import os
import pprint
import asyncio
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
from avnu_helpers import swap_eth_to_usdc, get_balance, swap_usdc_to_eth

load_dotenv(find_dotenv())

dev_private_key = os.environ.get("PRIVATE_KEY")
dev_public_key = os.environ.get("PUBLIC_KEY")
dry_run = os.environ.get("DRY_RUN")
starknet_rpc_url = os.environ.get("STARKNET_RPC_URL")

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

# Create Giza Model and Starknet Account instances
def create_model_and_account(
    model_id: int, version_id: int
):
    """
    Create a Giza model and a Starknet account.
    """

    # we create an instance of our Account
    node_url = starknet_rpc_url
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

# Make predictions
def predict(model: GizaModel, last_window_scaled: np.ndarray, scaler: MinMaxScaler):
    """
    Predict the ETH price.

    Args:
    last_window_scaled (np.ndarray): Input to the model.

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

async def execute_trading_strategy(model_id: int, version_id: int):
    """
    Execute the trading strategy based on the prediction.
    """
    logger = getLogger("agent_logger")
    eth_token_address = ADDRESSES["ETH"]["SN_MAIN"]
    usdc_token_address = ADDRESSES["USDC"]["SN_MAIN"]
    avnu_exchange = ADDRESSES["AVNU_EXCHANGE"]["SN_MAIN"]

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
        # get usdc balance , if > 0.2 , buy ETH with 0.1 USDC
        usdc_balance = await get_balance(account, usdc_token_address) 
        usdc_balance = usdc_balance / 10**6   
        logger.info(f"Your USDC balance: {usdc_balance:.2f}")
        if usdc_balance > 0.2:
            logger.info("Buying ETH with 0.1 USDC....")
            tx_hash = await swap_usdc_to_eth(account, 0.1)
            logger.info(f"Transaction hash: {tx_hash}")
        else:
            logger.info("Insufficient USDC balance to buy ETH")
        
    elif eth_price_prediction < current_eth_price * 0.99:
        eth_balance = await get_balance(account, eth_token_address)
        eth_balance = eth_balance / 10**18
        logger.info(f"ETH balance: {eth_balance:.18f}")
        if eth_balance > 0:
            logger.info("Selling ETH for USDC....")
            tx_hash = await swap_eth_to_usdc(account, eth_balance)
            logger.info(f"Transaction hash: {tx_hash}")
        else:
            logger.info("No ETH balance to sell")
    else:
        logger.info("No trading action needed at this time.")
   
    logger.info("-------Agent Execution Finished---------")

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

    asyncio.run(execute_trading_strategy( MODEL_ID, VERSION_ID))