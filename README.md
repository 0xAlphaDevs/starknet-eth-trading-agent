## Starknet Eth Trading Agent

On chain ETH trading strategy agent deployed on starknet mainnet | Built using Giza SDK

## Table of Contents

1. [All Links](#links)
2. [Instructions to Run ](#instructions-to-run)
3. [Project Overview](#project-overview)
4. [Tech Stack](#tech-stack)
5. [App Demo](#app-demo-screenshots)
6. [Team](#team)

## Links

- [Demo Video]()
- [Presentation]()

## Instructions to Setup

Welcome to this step-by-step tutorial for price prediction to assist in trading using the ETH/USDC pool. In this guide, we will walk through the entire process required to set up, deploy, and maintain an intelligent trading solution using the Giza stack. By the end of this tutorial, you will have a functional system capable of informing your trading decisions based on predictive market analysis.

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

**Funded Wallet**

You will need a funded Ethereum address linked to an Ape account. Follow the creating an account and funding the account parts of the MNIST tutorial to complete these steps.

**Environment Variables**

Create a .env file in the project directory and populate it with the following variables:
env

```
DEV_PASSPHRASE="<YOUR-APE-ACCOUNT-PASSWORD>"
SEPOLIA_RPC_URL="YOUR-RPC-URL"
```

We recommend using private RPCs but if you don't have one, use a public one like https://eth-sepolia.g.alchemy.com/v2/demo.

### 2. Building the Price Prediction Model

In this project, we are using a simple multi-layer perceptron to predict the next day's prices. After training the model, we need to compile it into the ONNX format, which will be used in the next step to transpile it into Cairo.

**Train the Model** :

The model will download the ETH/USDC prices, preprocess the data, train a simple neural network with Torch, and save the model in ONNX format.

```
Example script: model_training.py.
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

- **Fetch Price Data** : Retrieve current price data from the ETH/USDC pool on AVNU.
- **Predict Prices** : Use the deployed model to predict future prices based on historical data.

### 6. Defining the Execution Flow

Now we will use the giza-actions sdk to develop our AI Agent and adjust the LP position. We need to implement the following steps:

- Fetch all the required addresses
- Create the AI Agent instance
- Run verifiable inference
- Get the prediction value.
- Buy/Sell ETH/USDC according to the predicted value.

### 7. Running the AI Agent

Finally, we can execute our script with the desired parameters:

```
python action_agent.py --model-id <YOUR-MODEL-ID> --version-id <YOUR-VERSION-ID> (add input value if there is any)
```

## Project Overview

### Agent Business Case

The agent is designed to facilitate trading on the ETH/USDC pool using an AI-driven strategy. The goal is to optimize trades based on market conditions, leveraging machine learning models for prediction and decision-making.

### Functionality

- **Prediction Models** : Uses trained models to predict optimal trading times and amounts.
- **On-Chain Deployment** : Deployed on Starknet mainnet, ensuring transparency and immutability.

### Possible Improvements

- **Model Enhancement** : Continuous training with updated data to improve prediction accuracy.
- **User Interface** : Development of a user-friendly dashboard for monitoring and control.
- **Integration with DEXs** : Expanding support to decentralized exchanges for diversified trading opportunities.

## Tech Stack

- **Giza CLI & SDK** : For model training, ONNX conversion, transpilation and deployment.
- **Starknet Mainnet** : Blockchain platform for deploying and executing contracts.
- **yFinance** : Data source for fetching historical and real-time financial data.
- **Python pandas** : Data manipulation and analysis library.

## App Demo Screenshots

![image](/public/appDemo/1.png)
![image](/public/appDemo/2.jpg)
![image](/public/appDemo/3.jpg)
![image](/public/appDemo/4.jpg)
![image](/public/appDemo/5.jpg)
![image](/public/appDemo/6.jpg)

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
