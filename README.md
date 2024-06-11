## Starknet ETH Trading Agent

On chain ETH trading strategy agent deployed on starknet mainnet | Built using Giza SDK

## Table of Contents

1. [Links](#links)
2. [Project Overview](#project-overview)
3. [Project Documentation ](#project-documentation)
4. [Demo](#demo)
5. [Team](#team)

## Links

- [Demo Video]()
- [Presentation]()

## Project Overview

### Agent Business Case

The agent is designed to facilitate trading on the ETH/USDC pool using an AI-driven strategy. The goal is to optimize trades based on market conditions, leveraging machine learning models for prediction and decision-making.

### Tech Stack

- **Giza CLI & SDK** : For model training, ONNX conversion, cairo transpilation and deployment.
- **Starknet Mainnet** : Blockchain platform for deploying and executing contracts.
- **PyTorch** : Framework for building deep learning models.
- **Python Libraries** : Poetry for Python dependencies,Pandas, numpy, yFinance for historical data, scikit-learn for ML tooling.

### Possible Improvements

- **Model Enhancement** : Model training with more historical price data to improve prediction accuracy.
- **User Interface** : Development of a user-friendly dashboard for monitoring and controlling agent and integrated wallets for trading.
- **Dex Integrations and custom trading stragegies** : Expanding support to other decentralised exchanges and feature for building custom on-chain trading strategies and deploy them from UI with single click.

## Project Documentation

Welcome to this step-by-step tutorial for on-chain price prediction strategy to assist in trading on the ETH/USDC pool. In this guide, we will walk through the entire process required to setup, deploy, and run a AI trading agent using the Giza stack. By the end of this tutorial, you will have a functional system capable of executing trades on-chain based on ML model predictions.

### 1. Setting up Your Development Environment

### Install Required Tools

- Python 3.11 or later must be installed on your machine.
- Install `giza-sdk` to use Giza CLI and Giza agents:

```
  pip install giza-sdk
```

Additional libraries:

```
pip install -U torch pandas
```

**Create Giza Account**

If you don't have one, create a Giza account [here](https://docs.gizatech.xyz/products/platform/resources/users).

**Environment Variables**

Create a .env file in the project directory and populate it with the following variables:
env

```
DEV_PASSPHRASE="<YOUR-APE-ACCOUNT-PASSWORD>"
SEPOLIA_RPC_URL="YOUR-RPC-URL"
```

### 2. Building the Price Prediction Model

In this project, we are using a simple multi-layer perceptron to predict the next day's prices. After training the model, we need to compile it into the ONNX format, which will be used in the next step to transpile it into Cairo.

**Training the Model** :

The model will download the ETH/USDC prices, preprocess the data, train a simple neural network with Torch, and save the model in ONNX format.

```
Script: model_training.py.
```

### 3. Deploying Inference Endpoint

**_Login to Giza CLI_**

```
giza users login
```

**_Create Giza Workspace_**

If you don't have a workspace, create one with:

```
giza workspaces create
```

**_Create Giza Model_**

```
giza models create --name price-pred-with-zkml --description "Price prediction with ZKML"
```

Note the model-id.

**_Transpile Model to Cairo_**

```
giza transpile --model-id <YOUR-MODEL-ID> --framework CAIRO <PATH-TO-YOUR-ONNX-MODEL> --output-path <YOUR-OUTPUT-PATH>
```

**_Deploy Endpoint_**

```
giza endpoints deploy --model-id <YOUR-MODEL-ID> --version-id <YOUR-VERSION-ID>
```

### 4. Creating a Giza Agent

Next, we want to create a Giza Agent that executes the verifiable inference and interacts with the blockchain.

**_Create Agent_**

```
giza agents create --model-id <YOUR-MODEL-ID> --version-id <YOUR-VERSION-ID> --name <AGENT-NAME> --description <AGENT-DESCRIPTION>
```

Alternatively, if you have the endpoint-id:

```
giza agents create --endpoint-id <ENDPOINT-ID> --name <AGENT-NAME> --description <AGENT-DESCRIPTION>
```

### 5. Fetching and Predicting Prices

- **Fetch Price Data** : Retrieve current price data for ETH from yFinance for past 60 days.
- **Predict Prices** : Use the deployed model to predict future prices based on above data.

### 6. Defining the Execution Flow

Now we will use the giza-actions sdk to develop our AI Agent and execute trades on Starknet Mainnet. We need to implement the following steps:

- Load environment variables
- Create the AI Agent instance
- Run verifiable inference
- Get ETH prediction value.
- Buy/Sell ETH according to the predicted value.

### 7. Running the AI Agent

Finally, we can execute our script with the desired parameters:

```
python action_agent.py --model-id <YOUR-MODEL-ID> --version-id <YOUR-VERSION-ID>
```

## Demo

![image](/public/demo/1.png)
![image](/public/demo/2.jpg)
![image](/public/demo/3.jpg)
![image](/public/demo/4.jpg)
![image](/public/demo/5.jpg)
![image](/public/demo/6.jpg)

## Team

Team [AlphaDevs](https://www.alphadevs.dev) 👇

### Github

[Harsh Tyagi](https://github.com/mr-harshtyagi)
[Yashasvi Chaudhary](https://github.com/0xyshv)

### Twitter / X

[Harsh Tyagi](https://twitter.com/0xmht)
[Yashasvi Chaudhary](https://twitter.com/0xyshv)

## Thanks

- Feel free to reach out to the [AlphaDevs team](https://www.alphadevs.dev) with any questions or issues.

- We appreciate your interest in our project and welcome contributions and feature suggestions.
