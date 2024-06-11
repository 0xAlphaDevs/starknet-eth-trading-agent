
import requests
import os
from starknet_py.net.models import StarknetChainId
from starknet_py.net.account.account import Account
from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.contract import Contract
from starknet_py.hash.selector import get_selector_from_name
from starknet_py.net.client_models import Call

from addresses import ADDRESSES
from dotenv import find_dotenv, load_dotenv
from typing import List, Dict,Iterable

load_dotenv(find_dotenv())

taker_address = os.environ.get("PUBLIC_KEY")

# AVNU V3 API - STARKNET MAINNET

def get_quote(
    sell_token: str, buy_token: str, amount: float
):
    """
    Get a quote for a swap.
    """
    # we get the addresses of the tokens
    sell_token_address = ADDRESSES[sell_token]["SN_MAIN"]
    buy_token_address = ADDRESSES[buy_token]["SN_MAIN"]

    if sell_token == "ETH":
        sell_amount = hex(int(amount * 10 ** 18)) # 18 decimals for ETH
    elif buy_token == "ETH":
        sell_amount = hex(int(amount * 10 ** 6)) # 6 decimals for USDC

    # call AVNU V3 api to get quote - GET https://starknet.api.avnu.fi/swap/v2/quotes?
    url = f"https://starknet.api.avnu.fi/swap/v2/quotes?sellTokenAddress={sell_token_address}&buyTokenAddress={buy_token_address}&sellAmount={sell_amount}&size=1"

    response = requests.get(url)
    quote = response.json()
    return quote[0]

def build_swap_calldata(
    quote_id: str, taker_address: str,slippage: float,include_approve: bool
):
    """
    Build the calldata for a swap.
    """

    # call AVNU V3 api to build swap calldata - POST https://starknet.api.avnu.fi/swap/v2/build with request body
    url = f"https://starknet.api.avnu.fi/swap/v2/build"

    data = {
        "quoteId": quote_id,
        "takerAddress": taker_address,
        "slippage": slippage,
        "includeApprove": include_approve
    }

    response = requests.post(url, json=data)
    calldata = response.json()
    return calldata

# Starknet Blockchain Read/Write Functions

async def execute_transactions(
    calls: list, account: Account):
    """
    Execute transactions using Calls.
    """
    # prepare the calls
    all_calls = []
    for call in calls:
        prepared_call = Call(
            to_addr=call["to_addr"],
            selector=call["selector"],
            calldata=call["calldata"],
        )
        all_calls.append(prepared_call)

    # execute the calls
    resp = await account.execute_v3(calls=all_calls,auto_estimate=True)

    await account.client.wait_for_tx(resp.transaction_hash)

    return resp.transaction_hash

async def get_balance(
    account: Account, token_address: str
):
    """
    Get the balance of an account.
    """
    balance = await account.get_balance(token_address)
    return balance
    
async def swap_eth_to_usdc(
    account: Account, amount: int
):
    """
    Swap ETH to USDC.
    """
    try:
        quote = get_quote("ETH", "USDC", amount)
        # print(f"Quote: {quote}")
        calldata = build_swap_calldata(quote["quoteId"], taker_address, 0.05, True)
        # print(f"Calldata: {calldata}")
        calls = convert_to_call(calldata["calls"])
        # print(f"Calls: {calls}")
        tx_hash = await execute_transactions(calls, account)
        # print(f"Transaction hash: {hex(tx_hash)}")
        return hex(tx_hash)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

async def swap_usdc_to_eth(
    account: Account, amount: float
):
    """
    Swap USDC to ETH.
    """
    try:
        quote = get_quote("USDC", "ETH", amount)
        # print(f"Quote: {quote}")
        calldata = build_swap_calldata(quote["quoteId"], taker_address, 0.05, True)
        # print(f"Calldata: {calldata}")
        calls = convert_to_call(calldata["calls"])
        # print(f"Calls: {calls}")
        tx_hash = await execute_transactions(calls, account)
        # print(f"Transaction hash: {hex(tx_hash)}")
        return hex(tx_hash)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
# Helper Functions

def hex_to_int(hex_str):
    """Convert a hex string to an integer."""
    return int(hex_str, 16)

def convert_to_call(data):
    """Convert input data to a list of Call dataclass instances."""
    calls = []
    for item in data:
        to_addr = hex_to_int(item["contractAddress"])
        selector = get_selector_from_name(item["entrypoint"])  
        calldata = [hex_to_int(arg) for arg in item["calldata"]]
        calls.append({"to_addr" :to_addr, "selector":selector, "calldata":calldata})
    return calls