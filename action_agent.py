import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

PRIVATE_KEY = os.environ.get("PRIVATE_KEY")

print("private key is: ", PRIVATE_KEY)