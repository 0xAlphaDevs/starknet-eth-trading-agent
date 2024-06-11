from starknet_py.net.models import StarknetChainId
from starknet_py.net.account.account import Account
from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.contract import Contract

def check_allowance(
    account: Account, spender: str, token_address: str
):
    """
    Check the allowance of a spender.
    """
    # we create an instance of our Account
    node_url = "https://free-rpc.nethermind.io/mainnet-juno"
    client = FullNodeClient(node_url=node_url)

    # we create an instance of our Starknet contract
    # contract = StarknetContract(
    #     address=token_address,
    #     abi=abi,
    #     client=client,
    # )

    # # we call the check_allowance function
    # allowance = contract.check_allowance(
    #     account=account,
    #     spender=spender,
    # )
    # return allowance

def approve(
    account: Account, spender: str, token_address: str, amount: int
):
    """
    Approve a spender.
    """
    # we create an instance of our Account
    node_url = "https://free-rpc.nethermind.io/mainnet-juno"
    client = FullNodeClient(node_url=node_url)

    # we create an instance of our Starknet contract
    # contract = StarknetContract(
    #     address=token_address,
    #     abi=abi,
    #     client=client,
    # )

    # # we call the approve function
    # contract.approve(
    #     account=account,
    #     spender=spender,
    #     amount=amount,
    # )
    # return contract

def get_balance(
    account: Account, token_address: str
):
    """
    Get the balance of an account.
    """
    # we create an instance of our Account
    node_url = "https://free-rpc.nethermind.io/mainnet-juno"
    client = FullNodeClient(node_url=node_url)

    # we create an instance of our Starknet contract
    # contract = StarknetContract(
    #     address=token_address,
    #     abi=abi,
    #     client=client,
    # )

    # # we call the get_balance function
    # balance = contract.get_balance(
    #     account=account,
    # )
    # return balance

def swap_eth_to_usdc(
    account: Account, amount: int
):
    """
    Swap ETH to USDC.
    """
    # we create an instance of our Account
    node_url = "https://free-rpc.nethermind.io/mainnet-juno"
    client = FullNodeClient(node_url=node_url)

    # we create an instance of our Starknet contract
    # contract = StarknetContract(
    #     address=contract_address,
    #     abi=abi,
    #     client=client,
    # )

    # # we call the swap function
    # contract.multi_route_swap_function(
    #     account=account,
    #     amount=amount,
    # )
    # return contract

def swap_usdc_to_eth(
    account: Account, amount: int
):
    """
    Swap USDC to ETH.
    """
    # we create an instance of our Account
    node_url = "https://free-rpc.nethermind.io/mainnet-juno"
    client = FullNodeClient(node_url=node_url)

    # we create an instance of our Starknet contract
    # contract = StarknetContract(
    #     address=contract_address,
    #     abi=abi,
    #     client=client,
    # )

    # # we call the swap function
    # contract.multi_route_swap_function(
    #     account=account,
    #     amount=amount,
    # )
    # return contract
