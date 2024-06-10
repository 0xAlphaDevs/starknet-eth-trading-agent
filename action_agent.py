import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

dev_private_key = os.environ.get("PRIVATE_KEY")

print("private key is: ", dev_private_key)