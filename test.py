import subprocess
import os
import time
import json

# ==========================================
# CONFIGURATION
# ==========================================
# Paste broker URL here:
BROKER_URL = "amqps://ziveknnm:Qdcf42fxP7k19SkuwUZOAofH2S8d7oWO@raccoon.lmq.cloudamqp.com/ziveknnm"

# Automatically find the current project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DATA_FILE = os.path.join(BASE_DIR, "data", "h1.csv")
FINAL_MODEL_OUTPUT = os.path.join(BASE_DIR, "final_global_model.json")

def run_command(cmd, detach=False):
    """Runs a terminal command. If detach=True, runs in the background."""
    if detach:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, shell=True)

def cleanup_containers(num_clients):
    print("🧹 Cleaning up any old test containers...")
    # Dynamically generate the names of all potential client containers
    client_names = " ".join([f"test-client-{i}" for i in range(1, num_clients + 1)])
    run_command(f"docker rm -f test-server-local {client_names}", detach=False)

def main():
    print("\n" + "="*50)
    print("🚀 SAPPHIRE FEM - LOCAL ORCHESTRATOR SIMULATION")
    print("="*50 + "\n")

    # Read config dynamically
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        num_clients = config.get("network", {}).get("num_clients", 2)
        num_rounds = config.get("network", {}).get("num_rounds", 5)
        print(f"📖 Loaded config: {num_clients} clients, {num_rounds} rounds.")
    except Exception as e:
        print(f"⚠️ Could not read config.json. Defaulting to 2 clients. Error: {e}")
        num_clients = 2

    #  Clean up previous runs
    cleanup_containers(num_clients)

    #  Start the Server 
    print("\n🖥️ Starting Central Server...")
    server_cmd = (
        f'docker run --name test-server-local -d '
        f'-v "{CONFIG_FILE}":/app/config.json '
        f'fkabir0503/sapphire-fem:latest '
        f'python -m sapphire.server --broker-host "{BROKER_URL}" --output /tmp/final_global_model.json'
    )
    run_command(server_cmd, detach=False)

    time.sleep(2) # Give the server a moment to boot up

    # 1. Read config dynamically
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        num_clients = config.get("network", {}).get("num_clients", 2)
        num_rounds = config.get("network", {}).get("num_rounds", 5)
        
        # Fetch a forced global model if the config specifies one
        global_model = config.get("network", {}).get("model", "").strip()
        
        # Fetch and flatten all available models from model_groups
        model_groups = config.get("model_groups", {
            "grp1_linear": ["logistic"], 
            "grp2_neural": ["mlp"]
        })
        config_models = [m for models in model_groups.values() for m in models]
        
        print(f"📖 Loaded config: {num_clients} clients, {num_rounds} rounds.")
    except Exception as e:
        print(f"⚠️ Could not read config.json. Defaulting to fallbacks. Error: {e}")
        num_clients = 2
        global_model = ""
        config_models = ["mlp", "logistic"]

    

    for i in range(1, num_clients + 1):
        node_name = f"hospital{i}"
        container_name = f"test-client-{i}"
        
        # Select specific data file per client (data/h1.csv, data/h2.csv, etc.)
        client_data_candidate = os.path.join(BASE_DIR, "data", f"h{i}.csv")
        active_data_file = client_data_candidate if os.path.exists(client_data_candidate) else DATA_FILE

        # Select model (Force global_model if set, otherwise cycle through config_models)
        if global_model:
            assigned_model = global_model
        else:
            assigned_model = config_models[(i - 1) % len(config_models)]

        print(f"🏥 Starting {node_name} (Container: {container_name}) | Model: {assigned_model} | Data: {os.path.basename(active_data_file)}...")
        
        client_cmd = (
            f'docker run --name {container_name} --rm -d '
            f'-v "{active_data_file}":/app/dataset.csv '
            f'fkabir0503/sapphire-fem:latest '
            f'python -m sapphire.client --node {node_name} --data /app/dataset.csv --model {assigned_model} --broker-host "{BROKER_URL}"'
        )
        run_command(client_cmd, detach=False)

    # Attach to Server Logs 
    print("\n📡 Attaching to Server Logs (Waiting for training to finish)...\n")
    print("-" * 50)
    
    # This will block the script until the server container stops
    run_command("docker logs -f test-server-local")
    
    print("-" * 50)

    # Extract the model file
    print(f"\n💾 Extracting final model to: {FINAL_MODEL_OUTPUT}")
    run_command(f'docker cp test-server-local:/tmp/final_global_model.json "{FINAL_MODEL_OUTPUT}"')

    # 7. Final Cleanup
    cleanup_containers(num_clients)
    
    print("\n✅ Local simulation complete!")

if __name__ == "__main__":
    main()