import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

dev_private_key = os.environ.get("PRIVATE_KEY")
starknet_rpc_url = os.environ.get("STARKNET_RPC_URL")
dry_run = os.environ.get("DRY_RUN")

print("private key is: ", dev_private_key)
print("starknet rpc url is: ", starknet_rpc_url)
print("dry run is: ", dry_run)