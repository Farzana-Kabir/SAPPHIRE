import os
import sys
import time
import subprocess
import json
import argparse

# start rabbitmq container 
subprocess.run("docker start rabbitmq || docker run -d --restart unless-stopped --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:management", shell=True)



def parse_args():
    parser = argparse.ArgumentParser(description="Run SAPPHIRE Local FL Test Environment")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--target-col", default=None, help="Target label column override")
    parser.add_argument("--patient-id-col", default=None, help="Patient ID column override")
    return parser.parse_args()

args = parse_args()
base_dir = os.path.abspath(os.path.dirname(__file__))

config_path = os.path.join(base_dir, args.config)
try:
    with open(config_path, "r") as f:
        config = json.load(f)
    num_clients = config.get("network", {}).get("num_clients", 1)
    broker_url = config.get("network", {}).get("broker_url", "amqp://guest:guest@127.0.0.1:5672/")
except FileNotFoundError:
    print(f"⚠️ {config_path} not found. Defaulting to 1 client.")
    num_clients = 1
    broker_url = "amqp://guest:guest@127.0.0.1:5672/"

container_names = [f"test-client-{i}" for i in range(1, num_clients + 1)]
containers_str = " ".join(container_names)

print("🧹 Cleaning up old containers...")
os.system(f"docker rm -f {containers_str} >nul 2>&1")

client_broker_url = broker_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
sapphire_path = os.path.join(base_dir, "sapphire").replace('\\', '/')

# Build client optional arguments
extra_client_args = ""
if args.target_col:
    extra_client_args += f' --target-col "{args.target_col}"'
if args.patient_id_col:
    extra_client_args += f' --patient-id-col "{args.patient_id_col}"'

# START CLIENTS
for i in range(1, num_clients + 1):
    data_path = os.path.join(base_dir, f"data{i}").replace('\\', '/')
    container_name = f"test-client-{i}"
    node_name = f"hospital{i}"
    
    print(f"🏥 Starting Hospital {i} (Docker container: {container_name})...")
    run_cmd = (
        f'docker run -d --name {container_name} '
        f'-e PYTHONUNBUFFERED=1 '
        f'-e RABBITMQ_URL="{client_broker_url}" '
        f'-v "{data_path}:/app/data" '
        f'-v "{sapphire_path}:/app/sapphire" '
        f'sapphire-client:local '
        f'--node {node_name} --data /app/data{extra_client_args}'
    )
    os.system(run_cmd)

# AUTOMATED READINESS CHECK
print("\n⏳ Waiting for hospitals to load datasets and initialize...")
for container_name in container_names:
    ready = False
    while not ready:
        try:
            logs = subprocess.check_output(f"docker logs {container_name}", shell=True, text=True, stderr=subprocess.STDOUT)
            if "SAPPHIRE Node starting" in logs:
                ready = True
            elif "FATAL" in logs or "ERROR" in logs:
                print(f"❌ Initialization error in {container_name}:\n{logs}")
                sys.exit(1)
            else:
                time.sleep(2)
        except subprocess.CalledProcessError:
            time.sleep(2)

print("✅ All hospitals are online and ready!")

# START SERVER
print("🚀 Starting Central Server...")
base_env = os.environ.copy()
base_env["PYTHONPATH"] = base_dir
base_env["RABBITMQ_URL"] = broker_url

server_path = os.path.join(base_dir, "sapphire", "server.py")
server_cmd = [sys.executable, "-u", server_path, "--config", config_path]
server_process = subprocess.Popen(server_cmd, env=base_env)

print(f"\n📡 Network running!")

try:
    server_process.wait()
except KeyboardInterrupt:
    print("\n🛑 Manual interrupt detected. Shutting down...")
finally:
    server_process.terminate()
    #print("🧹 Removing Docker containers...")
    #os.system(f"docker rm -f {containers_str} >nul 2>&1")
    print("✅ All processes stopped gracefully.")