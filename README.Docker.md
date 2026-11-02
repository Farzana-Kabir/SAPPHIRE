# SAPPHIRE Federated Learning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](#)

**SAPPHIRE** is a privacy-preserving, heterogeneous federated learning framework designed for clinical data-sharing networks. Aligned with pan-European infrastructure standards (such as EUCAIM), SAPPHIRE enables top research institutions and hospitals to collaboratively train generalized predictive models without ever transferring sensitive patient data outside their local firewalls.

##  Key Features

* **Split Architecture:** Distinct, optimized Docker images for the central orchestrator and hospital clients to enforce the principle of least privilege.
* **Heterogeneous Model Support:** Clients can train entirely different local architectures (e.g., Multi-Layer Perceptrons, Random Forests, Logistic Regression). SAPPHIRE aligns them globally using advanced prototype alignment.
* **Secure Aggregation (SecAgg):** Cryptographic masking ensures the central server aggregates model updates without ever being able to inspect an individual hospital's raw model weights.
* **Highly Adaptable:** Seamlessly supports binary classification (e.g., disease diagnosis) and multi-class problems (e.g., disease intensity, staging, or risk levels) through a single, declarative JSON configuration.

##  System Architecture

SAPPHIRE utilizes a decentralized communication topology coordinated via a **RabbitMQ** message broker. 

1. **Server Image (`sapphire-server`):** Run by the Federated Experiments Manager (FEM). Responsible strictly for routing messages, coordinating rounds, performing Secure Aggregation, and saving the global model. It contains no data processing libraries.
2. **Client Image (`sapphire-client`):** Run by participating instituitons. Contains necessary libraries (PyTorch, Scikit-Learn, Pandas) to train models on local tabular or imaging features.

## 📋 Prerequisites

* **Docker:** Must be installed on all participating machines (Server and Clients).
* **RabbitMQ Broker:** A running AMQP broker (e.g., self-hosted RabbitMQ or CloudAMQP) accessible by all nodes via an `amqp://` or `amqps://` connection URL.

---

## ⚙️ 1. Orchestrator Configuration (`config.json`)

The FEM Administrator must define the experiment parameters in a `config.json` file. This acts as the single source of truth for the federated network.

SAPPHIRE is designed to be use-case agnostic. Below is an example configured for a **2-class cancer** prediction model, followed by a breakdown of the parameters:

```json
{
  "network": {
    "num_clients": 4,
    "num_rounds": 5,
    "model": ""
  },
  "model_groups": {
    "grp1_linear": ["logistic", "logistic_bn"],
    "grp2_neural": ["mlp", "transformer_lite"],
    "grp3_tree": ["decision_tree_soft", "rf"]
  },
  "labels": {
    "num_classes": 2,
    "target_columns": ["diagnosis", "target", "label", "condition", "class"],
    "class_names": { "0": "Negative/Benign", "1": "Positive/Malignant" },
    "label_mapping": {
    "malignant": 1,
    "positive": 1,
    "cancerous": 1,
    "benign": 0,
    "negative": 0,
    "healthy": 0,
    "0": 0,
    "1": 1
  }
  },
}
```

### Parameter Breakdown

#### `network`
* **`num_clients`**: The exact number of hospital nodes the server must wait for before beginning the first training round. 
* **`num_rounds`**: The total number of federated training cycles (global epochs) the network will execute before saving the final model and shutting down.
* **`model`**: (Optional) Leave blank to allow heterogeneous training. If populated (e.g., `"mlp"`), forces all nodes to train using that specific architecture.

#### `model_groups`
Defines the architectural families available for heterogeneous training. Nodes can select any model listed here, and SAPPHIRE will automatically group and align their prototype vectors based on these categories.

#### `labels`
* **`num_classes`**: Defines the shape of the prediction. Set to `2` for binary classification (e.g., healthy vs. sick) or `>2` for multi-class problems (e.g., stages 1, 2, 3, 4).
* **`target_column`**: The default column name SAPPHIRE will look for inside the hospitals' CSV files to find the labels. 
* **`label_mapping`**: A dictionary used to standardize data dynamically. If a hospital's raw CSV contains text like `"mild"` or `"severe"`, SAPPHIRE automatically converts them to the integers `0` and `2` before training begins.

---

## 🚀 2. Deploying the Central Server

The server requires the `config.json` file and a designated `output` folder to save the final global model. 

Run the following command on the orchestrator machine:

```bash
docker run -d \
  --name sapphire-server \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/output:/app/output \
  registry.gitlab.bsc.es/fkabir/sapphire-server:latest \
  python -m sapphire.server \
  --config /app/config.json \
  --broker-host "amqps://BROKER_URL"
```

To monitor the aggregation process in real-time, view the server logs:
```bash
docker logs -f sapphire-server
```

---

## 🏥 3. Deploying a Client Node (Hospitals)

Provide the participating institutions with the RabbitMQ broker URL and the following execution instructions.

Client nodes must mount their local dataset (e.g., CSV format) into the container. SAPPHIRE supports heterogeneous databases: if a hospital's database uses a different column name for the target labels (e.g., `local_status` instead of the global `disease_severity`), they can dynamically map it using the `--target-col` flag without modifying their raw data.

**Client Execution Command:**
```bash
docker run -d \
  --name sapphire-node \
  -v /path/to/local/data.csv:/app/dataset.csv \
  registry.gitlab.bsc.es/fkabir/sapphire-client:latest \
  --node unique_hospital_id \
  --data /app/dataset.csv \
  --model mlp \
  --target-col local_status \
  --broker-host "amqps://BROKER_URL"
```

* **`--node`**: A unique identifier for the hospital (e.g., `hospital_1`).
* **`--model`**: The local architecture to train (e.g., `mlp`, `rf`, `logistic`).
* **`--target-col`**: (Optional) Overrides the target column name locally.

---

## 💾 4. Retrieving the Global Model

Upon the completion of the defined `num_rounds`, the central server automatically broadcasts a shutdown signal to all connected clients.

The server will perform the final Secure Aggregation computation and output the global model weights to the mounted directory. You can find the resulting artifact at:
`./output/final_global_model.json`