import argparse
import numpy as np
import logging
import threading
import hashlib
import time
import secrets
import sys
import json
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sapphire.broker import FLMessageBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("pika").setLevel(logging.WARNING)
SECAGG_MODULUS = 2 ** 256
QUANTIZATION_SCALE = 100000
HALF_MODULUS = SECAGG_MODULUS // 2

def get_safe_prime():
    hex_prime = (
        "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
        "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
        "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
        "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
        "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
        "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
        "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
        "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
        "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
        "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
        "15728E5A8AACAA68FFFFFFFFFFFFFFFF"
    )
    return int(hex_prime, 16), 2

def generate_aes_ctr_mask(seed_bytes, session_id, round_num, size):
    nonce = hashlib.sha256(f"{session_id}:{round_num}".encode()).digest()[:16]
    cipher = Cipher(algorithms.AES(seed_bytes), modes.CTR(nonce))
    raw_bytes = cipher.encryptor().update(b'\x00' * (size * 32))

    uint64_chunks = np.frombuffer(raw_bytes, dtype=np.uint64).reshape(-1, 4)
    mask_ints = [
        (int(a) << 192) | (int(b) << 128) | (int(c) << 64) | int(d)
        for a, b, c, d in uint64_chunks
    ]
    return np.array(mask_ints, dtype=object) % SECAGG_MODULUS

def lagrange_interpolate(shares, safe_prime):
    while True:
        try:
            shares_snapshot = {int(k): int(v) for k, v in shares.items()}
            break
        except RuntimeError:
            pass
    secret = 0
    for xi, yi in shares_snapshot.items():
        num, den = 1, 1
        for xj in shares_snapshot.keys():
            if xi != xj:
                num = (num * -xj) % safe_prime
                den = (den * (xi - xj)) % safe_prime
        secret = (secret + yi * num * pow(den, safe_prime - 2, safe_prime)) % safe_prime
    return secret

class OrchestratorServer:
    def __init__(self, broker_host="localhost", config_path="config.json", output_path="/app/output/final_global_model.json",
                 num_clients=None, num_rounds=None):
        
        # Load fallback values from config.json if present
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

        _broker_cfg = config.get("message_broker", {})
        self.model_groups = config.get("model_groups", {
            "grp1_linear": ["logistic", "logistic_bn"],
            "grp2_neural": ["mlp", "transformer_lite"],
            "grp3_tree": ["decision_tree_soft", "rf"]
        })
        self.global_model = config.get("network", {}).get("model", "").strip()
        self.num_classes = config.get("labels", {}).get("num_classes", 2)

        # CLI overrides take precedence over config.json
        self.expected_clients = num_clients or config.get("network", {}).get("num_clients", 2)
        self.num_rounds = num_rounds or config.get("network", {}).get("num_rounds", 5)
        self.output_path = output_path

        self.broker = FLMessageBroker(
            host=broker_host,
            queue_name=_broker_cfg.get("queue_name", "fl_server_queue_v2"),
            exchange_name=_broker_cfg.get("exchange_name", "fl_broadcast_exchange"),
            heartbeat=_broker_cfg.get("heartbeat_seconds", 3600),
            blocked_connection_timeout=_broker_cfg.get("blocked_connection_timeout_seconds", 3600),
        )
        self.history = []
        self.unmask_event = threading.Event()
        self.session_id = secrets.randbits(64)
        self.em_d = 0
        self.vector_size = 0
        self.SAFE_PRIME = None
        self.GENERATOR_G = None
        self.momentum = 0.7

        self.client_id_map = {}
        self.next_id = 1
        self._state_lock = threading.Lock()

        self.client_keys = {}
        self.client_comm_keys = {}
        self.clients_sent_shares = set()

        self.shares_map = {}
        self.client_num_classes = {}
        self.round_updates = []
        self._seen_update_ids = set()
        self.collected_unmask_shares = {"loc": {}, "priv": {}}
        self._current_unmask_round = -1
        self.current_round = 0

    def handle_messages(self, msg):
        m_type = msg.get("type") or msg.get("command")

        if m_type == "REGISTER_REPLY":
            node_name = msg["node_name"]
            with self._state_lock:
                if node_name not in self.client_id_map:
                    self.client_id_map[node_name] = self.next_id
                    self.next_id += 1

                assigned_id = self.client_id_map[node_name]
                self.client_num_classes.setdefault(assigned_id, self.num_classes)

        elif m_type == "PUB_KEY":
            c_id = str(msg["id"])
            with self._state_lock:
                self.client_keys[c_id] = str(msg["key"])
                self.client_comm_keys[c_id] = str(msg["comm_pub"])

        elif m_type == "ENCRYPTED_SHARES":
            with self._state_lock:
                self.clients_sent_shares.add(str(msg["id"]))
                for dest_id, payload in msg["shares"].items():
                    self.shares_map.setdefault(str(dest_id), {})[str(msg["id"])] = payload

        elif m_type == "UNMASK_REPLY":
            reply_round = msg.get("round")
            if reply_round != self._current_unmask_round:
                return
            with self._state_lock:
                for s_id, sh in msg["shares"].get("loc", {}).items():
                    self.collected_unmask_shares.setdefault("loc", {}).setdefault(int(s_id), {})[str(msg["id"])] = str(sh)
                for d_id, sh in msg["shares"].get("priv", {}).items():
                    self.collected_unmask_shares.setdefault("priv", {}).setdefault(int(d_id), {})[str(msg["id"])] = str(sh)
                threshold = (self.expected_clients // 2) + 1
                loc_done = all(len(s) >= threshold for s in self.collected_unmask_shares["loc"].values())
                priv_done = all(len(s) >= threshold for s in self.collected_unmask_shares["priv"].values())
                if loc_done and priv_done:
                    self.unmask_event.set()

        elif m_type == "UPDATE":
            client_id = int(msg["client_id"])
            msg_round = msg.get("round")
            with self._state_lock:
                dedup_key = (client_id, msg_round)
                if dedup_key in self._seen_update_ids:
                    return
                self._seen_update_ids.add(dedup_key)
                self.round_updates.append(msg)

    def run_autopilot(self):
        print("\n" + "=" * 50 + "\n  EUCAIM SAPPHIRE SERVER STARTING\n" + "=" * 50)

        try:
            conn = self.broker._get_connection()
            ch = conn.channel()
            ch.queue_delete(queue='fl_server_queue_v2')
            queue_args = {
                "x-max-length": 10000,
                "x-max-length-bytes": 52428800,
                "x-overflow": "drop-head"
            }
            ch.queue_declare(queue='fl_server_queue_v2', durable=True, arguments=queue_args)
            ch.queue_purge(queue='fl_server_queue_v2')
            conn.close()
        except Exception:
            pass

        threading.Thread(target=self.broker.listen_client_queues,
                         args=(self.handle_messages,), daemon=True).start()

        wait_time = 0
        while True:
            with self._state_lock:
                n_registered = len(self.client_num_classes)
            if n_registered >= self.expected_clients:
                break

            self.broker.broadcast({
                "command": "SETUP",
                "num_classes": self.num_classes,
                "model_groups": self.model_groups,
                "global_model": self.global_model
            })
            time.sleep(3)
            wait_time += 3
            if wait_time >= 600:
                print("❌ ERROR: Timed out waiting for hospital nodes to register.")
                sys.exit(1)

        self.em_d = max(self.num_classes * 8, 16)
        self.vector_size = self.em_d * self.num_classes
        self.SAFE_PRIME, self.GENERATOR_G = get_safe_prime()

        print(f" Registered {n_registered}/{self.expected_clients} nodes. Session initialized.")

        self.broker.broadcast({
            "command": "INIT",
            "session_id": str(self.session_id),
            "prime": str(self.SAFE_PRIME),
            "generator": str(self.GENERATOR_G),
            "em_d": self.em_d,
            "vector_size": self.vector_size,
            "num_class": self.num_classes,
            "total_clients": self.expected_clients,
            "num_rounds": self.num_rounds,
            "client_map": self.client_id_map
        })

        while True:
            with self._state_lock:
                n_keys = len(self.client_keys)
            if n_keys >= self.expected_clients:
                break
            time.sleep(1)

        with self._state_lock:
            keys_snap = dict(self.client_keys)
            comm_keys_snap = dict(self.client_comm_keys)

        self.broker.broadcast({
            "command": "KEY_DIRECTORY",
            "keys": keys_snap,
            "comm_keys": comm_keys_snap
        })

        while True:
            with self._state_lock:
                n_shares = len(self.clients_sent_shares)
            if n_shares >= self.expected_clients:
                break
            time.sleep(1)

        with self._state_lock:
            shares_snapshot = dict(self.shares_map)

        self.broker.broadcast({"command": "START", "shares_map": shares_snapshot})

    def run_federated_loop(self):
        start_time = time.perf_counter()
        self.global_prototypes = None

        for current_round in range(1, self.num_rounds + 1):
            print(f"\n ROUND {current_round}/{self.num_rounds} STARTING")
            with self._state_lock:
                self.round_updates = []
                self._seen_update_ids = set()

            global_protos_list = self.global_prototypes.tolist() if self.global_prototypes is not None else []
            self.broker.broadcast({
                "command": "TRAIN_ROUND",
                "round": current_round,
                "global_prototypes": global_protos_list
            })

            max_wait_time = 60
            wait_time = 0
            while wait_time < max_wait_time:
                with self._state_lock:
                    n_updates = len(self.round_updates)
                if n_updates == self.expected_clients:
                    break
                time.sleep(1)
                wait_time += 1

            self.aggregate_fit(current_round)

        print(f"\n✅ FINAL SESSION COMPLETE — Runtime: {time.perf_counter() - start_time:.2f}s")

        # Save to orchestrator-designated output path
        if self.global_prototypes is not None:
            out_dir = os.path.dirname(os.path.abspath(self.output_path))
            os.makedirs(out_dir, exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump({"prototypes": self.global_prototypes.tolist()}, f, indent=2)
            print(f"💾 Final global model saved to '{self.output_path}'")

        # Signal completion to client nodes so they can exit gracefully
        self.broker.broadcast({"command": "SESSION_COMPLETE"})
        time.sleep(2)

    def trigger_unmasking(self, survivors, dropped, current_round):
        self._current_unmask_round = current_round
        with self._state_lock:
            self.collected_unmask_shares = {
                "loc": {s: {} for s in survivors},
                "priv": {d: {} for d in dropped}
            }
        self.unmask_event.clear()
        self.broker.broadcast({
            "command": "UNMASK",
            "survivors": survivors,
            "dropped": dropped,
            "round": current_round
        })
        self.unmask_event.wait(timeout=120)
        with self._state_lock:
            return dict(self.collected_unmask_shares)

    def aggregate_fit(self, current_round):
        with self._state_lock:
            updates = list(self.round_updates)

        n_updates = len(updates)
        threshold = (self.expected_clients // 2) + 1

        if n_updates < threshold:
            print(f"\n⚠️ ABORTING ROUND {current_round}: Only {n_updates} survived. Threshold={threshold}.")
            return

        masked_arrays = [np.array([int(x) for x in msg["masked_data"]], dtype=object) for msg in updates]
        summed_masked = np.sum(np.array(masked_arrays, dtype=object), axis=0) % SECAGG_MODULUS

        survivors = [int(msg["client_id"]) for msg in updates]
        with self._state_lock:
            all_ids = set(int(k) for k in self.client_keys.keys())
        dropped = sorted(list(all_ids - set(survivors)))

        mask_size = self.vector_size + self.num_classes + 3
        total_recon_mask = np.zeros(mask_size, dtype=object)

        unmask_data = self.trigger_unmasking(survivors, dropped, current_round)

        with self._state_lock:
            client_keys_snapshot = dict(self.client_keys)

        for s_id in survivors:
            shares = unmask_data.get("loc", {}).get(s_id, {})
            if len(shares) < threshold:
                continue
            seed_val = lagrange_interpolate(shares, self.SAFE_PRIME)
            seed_bytes = seed_val.to_bytes(32, 'big')
            total_recon_mask = (total_recon_mask +
                                generate_aes_ctr_mask(seed_bytes, self.session_id,
                                                      current_round, mask_size)) % SECAGG_MODULUS

        for d_id in dropped:
            shares = unmask_data.get("priv", {}).get(d_id, {})
            if len(shares) < threshold:
                continue
            priv_k = lagrange_interpolate(shares, self.SAFE_PRIME)
            for s_id in survivors:
                survivor_pub = client_keys_snapshot[str(s_id)]
                shared = pow(int(survivor_pub), priv_k, self.SAFE_PRIME)
                ss_bytes = shared.to_bytes(max(32, (shared.bit_length() + 7) // 8), 'big')
                hkdf = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=str(self.session_id).encode(),
                    info=b'secagg_masking_seed'
                )
                seed = hkdf.derive(ss_bytes)
                mask = generate_aes_ctr_mask(seed, self.session_id, current_round, mask_size)
                total_recon_mask = ((total_recon_mask + mask) % SECAGG_MODULUS) if s_id > d_id \
                    else ((total_recon_mask - mask) % SECAGG_MODULUS)

        final_sum = (summed_masked - total_recon_mask) % SECAGG_MODULUS
        final_sum_list = [int(x) for x in final_sum]

        vector_size = self.vector_size
        num_classes = self.num_classes

        prototype_ints = final_sum_list[:vector_size]
        presence_ints = final_sum_list[vector_size:vector_size + num_classes]
        summed_acc = final_sum_list[vector_size + num_classes]
        summed_bal_acc = final_sum_list[vector_size + num_classes + 1]
        total_weight = final_sum_list[vector_size + num_classes + 2]

        if total_weight > 0:
            mean_acc = (summed_acc / total_weight) / QUANTIZATION_SCALE * 100.0
            mean_bal = (summed_bal_acc / total_weight) / QUANTIZATION_SCALE * 100.0
        else:
            mean_acc = mean_bal = 0.0

        summed_prototypes = np.array(prototype_ints, dtype=np.float64).reshape(num_classes, vector_size // num_classes)
        safe_counts = np.maximum(np.array(presence_ints, dtype=np.float64), 1.0).reshape(num_classes, 1)

        np_mat = summed_prototypes / (safe_counts * QUANTIZATION_SCALE)
        np_mat = np.nan_to_num(np_mat, nan=0.0, posinf=0.0, neginf=0.0)

        norm = np.linalg.norm(np_mat, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        new_prototypes = (np_mat / norm).flatten()

        if current_round <= 2:
            self.global_prototypes = new_prototypes
        else:
            self.global_prototypes = (self.momentum * self.global_prototypes) + ((1.0 - self.momentum) * new_prototypes)

        gp_mat = self.global_prototypes.reshape(num_classes, -1)
        norm = np.linalg.norm(gp_mat, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        self.global_prototypes = (gp_mat / norm).flatten()

        print(f"\n✅ SECURE AGGREGATION SUCCESSFUL (Round {current_round})")
        print(f"🌍 Global Model Update (first 5 vals): {self.global_prototypes[:5]}")
        print(f"📊 ROUND {current_round} RESULTS | global average accuracy: {mean_acc:.2f}%  balanced: {mean_bal:.2f}%\n")

def main():
    parser = argparse.ArgumentParser(description="SAPPHIRE EUCAIM Central Aggregator")
    parser.add_argument("--broker-host", default="localhost", help="RabbitMQ broker host or amqps:// URL")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--output", default="/app/output/final_global_model.json", help="Destination path for final model")
    parser.add_argument("--clients", type=int, default=None, help="Override number of expected clients")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of training rounds")
    parser.add_argument("--momentum", type=float, default=0.7)
    args = parser.parse_args()

    server = OrchestratorServer(
        broker_host=args.broker_host,
        config_path=args.config,
        output_path=args.output,
        num_clients=args.clients,
        num_rounds=args.rounds
    )
    server.momentum = args.momentum

    print(f" Loaded configuration: Expecting {server.expected_clients} clients for {server.num_rounds} rounds.")
    server.run_autopilot()
    server.run_federated_loop()
    print("🏁 Orchestrator server execution completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()