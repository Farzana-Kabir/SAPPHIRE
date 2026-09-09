import argparse, secrets, hashlib, threading, os, random, json, time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sapphire.broker import FLMessageBroker
from sapphire.fedproto import build_local_model, train_and_extract_prototypes, GROUP_MAP

QUANTIZATION_SCALE, SECAGG_MODULUS = 100000, 2**256

def generate_shares(secret, n, t, modulus):
    coeffs = [secret] + [secrets.randbelow(modulus) for _ in range(t - 1)]
    return {i: sum(c * pow(i, e, modulus) for e, c in enumerate(coeffs)) % modulus
            for i in range(1, n + 1)}

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

def self_recognize_and_load(node_type, data_path=None):
    """
    Loads dataset dynamically. Supports both direct file paths and mounted directories.
    """
    if data_path is None:
        data_dir = os.environ.get("DATA_DIR", "./data")
        data_path = os.path.join(data_dir, f"{node_type}.csv")

    # If the orchestrator mounted a directory, find the target CSV inside
    if os.path.isdir(data_path):
        candidates = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        if not candidates:
            raise FileNotFoundError(f"❌ ERROR: No .csv file found in data directory: {data_path}")
        data_path = candidates[0]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ ERROR: Cannot find dataset at {data_path}")

    print(f"📊 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path).dropna(how='all')

    if 'label' not in df.columns:
        raise ValueError(
            f"❌ ERROR: {data_path} has no 'label' column. Every client's data must "
            f"conform to canonical form (numeric feature columns + a 'label' column)."
        )

    if 'split' in df.columns:
        df_train = df[df['split'] == 'train']
        df_test  = df[df['split'] == 'test']

        X_train_raw = df_train.drop(columns=['label', 'split'], errors='ignore').select_dtypes(include=[np.number]).values
        y_train = df_train['label'].values.astype(np.int64)
        X_test_raw  = df_test.drop(columns=['label', 'split'], errors='ignore').select_dtypes(include=[np.number]).values
        y_test  = df_test['label'].values.astype(np.int64)

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)
    else:
        X = df.drop(columns=['label'], errors='ignore').select_dtypes(include=[np.number]).values
        y_int = df['label'].values.astype(np.int64)
        X_scaled = StandardScaler().fit_transform(X).astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_int, test_size=0.2, random_state=42, stratify=y_int)

    return X_train, X_test, y_train, y_test

def _predict(model, X_test, num_class, em_d, parameters, has_global_protos, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.as_tensor(X_test, dtype=torch.float32).to(device)
        v, logits = model(X_t)

        if has_global_protos and len(parameters) > 0:
            v_norm     = F.normalize(v, dim=1)
            gp_tensor  = torch.tensor(parameters[0], dtype=torch.float32).reshape(num_class, em_d).to(device)
            gp_norm    = F.normalize(gp_tensor, dim=1)
            similarity = torch.mm(v_norm, gp_norm.T)

            cos_probs   = F.softmax(similarity / 0.5, dim=1)
            logit_probs = F.softmax(logits, dim=1)
            ensemble    = 0.5 * cos_probs + 0.5 * logit_probs
            return torch.argmax(ensemble, dim=1).cpu().numpy()
        else:
            return torch.argmax(logits, dim=1).cpu().numpy()

class InteractiveNode:
    def __init__(self, node_type, requested_model, X_train, y_train,
                 X_test, y_test, broker_host="localhost"):
        
        self.broker = FLMessageBroker(host=broker_host)
        self.node_type   = node_type
        self.X_train, self.y_train = X_train, y_train
        self.X_test,  self.y_test  = X_test,  y_test
        self.requested_model = requested_model
        self.start_event     = threading.Event()
        self.finish_event    = threading.Event()
        self.my_id           = -1
        self.registered      = False
        self.seeds           = {}
        self.sess_id         = -1
        self._train_lock     = threading.Lock()

    def handle_messages(self, msg):
        cmd = msg.get("command") or msg.get("type")

        if cmd == "SETUP":
            if not self.registered:
                self.num_class = int(msg["num_classes"])
                self.model_groups = msg.get("model_groups", {})
                if not self.requested_model:
                    self.requested_model = msg.get("global_model", "")
                
                seen_classes = np.unique(np.concatenate([self.y_train, self.y_test]))
                if seen_classes.max(initial=-1) >= self.num_class or seen_classes.min(initial=0) < 0:
                    print(f"❌ FATAL ERROR: Local dataset class indices exceed server num_classes={self.num_class}")
                    sys.exit(1)

                self.broker.send_to_server({
                    "type": "REGISTER_REPLY",
                    "node_name": self.node_type
                })

        elif cmd == "INIT":
            if getattr(self, 'priv_k', None) is not None:
                return 

            self.my_id = msg["client_map"][self.node_type]
            self.registered = True
            self.sess_id, self.p, self.g = int(msg["session_id"]), int(msg["prime"]), int(msg["generator"])
            self.em_d, self.total_clients = int(msg["em_d"]), int(msg["total_clients"])
            
            self.priv_k = secrets.randbelow(self.p)
            self.num_rounds = int(msg.get("num_rounds", 10))
            self.local_seeds = [secrets.randbits(256) for _ in range(self.num_rounds)]
            self.comm_priv = secrets.randbelow(self.p)

            self.broker.send_to_server({
                "type": "PUB_KEY", 
                "id": self.my_id,
                "key": str(pow(self.g, self.priv_k, self.p)),
                "comm_pub": str(pow(self.g, self.comm_priv, self.p))
            })

        elif cmd == "KEY_DIRECTORY":
            if getattr(self, 'peer_keys', None) is not None:
                return

            self.peer_keys = msg["keys"]           
            self.peer_comm_keys = msg["comm_keys"] 
            threshold = (self.total_clients // 2) + 1
            
            priv_shares = generate_shares(self.priv_k, self.total_clients, threshold, self.p)
            all_local_shares = [generate_shares(seed, self.total_clients, threshold, self.p) for seed in self.local_seeds]

            my_encrypted_shares = {}
            for peer_id_str, peer_comm_pub_str in self.peer_comm_keys.items():
                peer_id = int(peer_id_str)
                shared_secret = pow(int(peer_comm_pub_str), self.comm_priv, self.p)
                ss_bytes = shared_secret.to_bytes(max(32, (shared_secret.bit_length() + 7) // 8), 'big')

                hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'secagg_encryption')
                aes_key = hkdf.derive(ss_bytes)
                aesgcm = AESGCM(aes_key)
                nonce = secrets.token_bytes(12)
                
                peer_loc_shares_list = [str(round_shares[peer_id]) for round_shares in all_local_shares]
                payload = json.dumps({"priv": str(priv_shares[peer_id]), "loc": peer_loc_shares_list}).encode()
                ciphertext = aesgcm.encrypt(nonce, payload, None)

                my_encrypted_shares[peer_id_str] = {
                    "nonce": nonce.hex(),
                    "ciphertext": ciphertext.hex()
                }

            self.broker.send_to_server({
                "type": "ENCRYPTED_SHARES",
                "id": self.my_id,
                "shares": my_encrypted_shares
            })

        elif cmd == "START":
            my_inbound_shares = msg.get("shares_map", {}).get(str(self.my_id), {})
            self.stored_shares = {}

            for sender_id_str, enc_payload in my_inbound_shares.items():
                sender_comm_pub = int(self.peer_comm_keys[sender_id_str])
                shared_secret = pow(sender_comm_pub, self.comm_priv, self.p)
                ss_bytes = shared_secret.to_bytes(max(32, (shared_secret.bit_length() + 7) // 8), 'big')
                hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'secagg_encryption')
                aes_key = hkdf.derive(ss_bytes)

                aesgcm = AESGCM(aes_key)
                nonce = bytes.fromhex(enc_payload["nonce"])
                ciphertext = bytes.fromhex(enc_payload["ciphertext"])
                
                try:
                    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
                    self.stored_shares[sender_id_str] = json.loads(plaintext.decode())
                except Exception as e:
                    print(f"Decryption failed for shares from {sender_id_str}: {e}")

            for o_id, pk in self.peer_keys.items():
                if int(o_id) != self.my_id:
                    shared = pow(int(pk), self.priv_k, self.p)
                    ss_bytes = shared.to_bytes(max(32, (shared.bit_length() + 7) // 8), 'big')
                    hkdf = HKDF(
                        algorithm=hashes.SHA256(), 
                        length=32, 
                        salt=str(self.sess_id).encode(), 
                        info=b'secagg_masking_seed'
                    )
                    self.seeds[o_id] = hkdf.derive(ss_bytes)
            
            self.start_event.set()

        elif cmd == "UNMASK":
            rnd = msg.get("round")
            reply_shares = {"loc": {}, "priv": {}}
            for s in msg.get("survivors", []):
                if str(s) in self.stored_shares and "loc" in self.stored_shares[str(s)]:
                    reply_shares["loc"][str(s)] = self.stored_shares[str(s)]["loc"][rnd - 1]
            for d in msg.get("dropped", []):
                if str(d) in self.stored_shares and "priv" in self.stored_shares[str(d)]:
                    reply_shares["priv"][str(d)] = self.stored_shares[str(d)]["priv"]
            self.broker.send_to_server({
                "type": "UNMASK_REPLY",
                "id": self.my_id,
                "round": rnd,
                "shares": reply_shares
            })

        elif cmd == "TRAIN_ROUND":
            threading.Thread(target=self._train_round, args=(msg,), daemon=True).start()

        elif cmd == "SESSION_COMPLETE":
            print("✅ Received SESSION_COMPLETE from server. Shutting down cleanly.")
            self.finish_event.set()

    def _train_round(self, msg):
        with self._train_lock:
            while getattr(self, 'model', None) is None or getattr(self, 'optimizer', None) is None:
                time.sleep(0.1)

            rnd                = msg["round"]
            global_protos_list = msg.get("global_prototypes", [])
            has_global_protos  = len(global_protos_list) > 0
            parameters         = [np.array(global_protos_list, dtype=np.float32)] if has_global_protos else []
            device             = next(self.model.parameters()).device

            current_lr = 0.002 * (0.95 ** (rnd - 1))
            for pg in self.optimizer.param_groups:
                pg['lr'] = current_lr
                
            v_i = train_and_extract_prototypes(
                self.model,
                self.optimizer,
                parameters,
                self.X_train,
                self.y_train,
                self.em_d,
                self.num_class,
                current_round=rnd
            )

            preds = _predict(self.model, self.X_test, self.num_class, self.em_d,
                             parameters, has_global_protos, device)
            acc          = float(accuracy_score(self.y_test, preds))
            balanced_acc = float(balanced_accuracy_score(self.y_test, preds))

            scaled_v_i = []
            v_i_reshaped = v_i.reshape(self.num_class, self.em_d)
            for c in range(self.num_class):
                for x in v_i_reshaped[c]:
                    val = 0.0 if (np.isnan(x) or np.isinf(x)) else x
                    scaled_v_i.append(int(round(val * QUANTIZATION_SCALE)))
            
            class_presence = [1 if c in self.y_train else 0 for c in range(self.num_class)]
            scaled_v_i.extend(class_presence)
            
            local_weight = len(self.y_train)
            scaled_v_i.append(int(round(acc * local_weight * QUANTIZATION_SCALE)))
            scaled_v_i.append(int(round(balanced_acc * local_weight * QUANTIZATION_SCALE)))
            scaled_v_i.append(local_weight)
            
            scaled_v_i = np.array(scaled_v_i, dtype=object)
            mask_size  = len(scaled_v_i)
            
            p_mask = np.zeros(mask_size, dtype=object)
            for o_id, seed in self.seeds.items():
                mask = generate_aes_ctr_mask(seed, self.sess_id, rnd, mask_size)
                p_mask = (p_mask + mask) % SECAGG_MODULUS if self.my_id > int(o_id) \
                    else (p_mask - mask) % SECAGG_MODULUS

            current_seed = self.local_seeds[rnd - 1]
            current_seed_bytes = current_seed.to_bytes(32, 'big')
            
            combined_mask = (p_mask + generate_aes_ctr_mask(current_seed_bytes,
                                                             self.sess_id, rnd, mask_size)) % SECAGG_MODULUS
            y_i_str = [str(x) for x in (scaled_v_i + combined_mask) % SECAGG_MODULUS]
            
            self.broker.send_to_server({
                "command":      "UPDATE",
                "client_id":    self.my_id,
                "round":        rnd,
                "masked_data":  y_i_str
            })

    def start(self):
        print(f"🚀 SAPPHIRE Node starting: {self.node_type}")
        threading.Thread(target=lambda: self.broker.listen_broadcasts(self.handle_messages),
                         daemon=True).start()
        self.start_event.wait()

        model_to_group = {m: g for g, models in self.model_groups.items() for m in models}
        if self.requested_model and self.requested_model in model_to_group:
            self.model_name = self.requested_model
        else:
            self.model_name = random.choice(list(model_to_group.keys()))

        self.group = model_to_group[self.model_name]
        print(f" Node configured: model={self.model_name} (family={self.group})")

        self.model, _  = build_local_model(self.group, self.model_name,
                                            self.X_train, self.y_train,
                                            self.em_d, self.num_class)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002, weight_decay=1e-4)

        # Block until the federated training completes or times out
        self.finish_event.wait(timeout=3600)
        print(f"🏁 Node {self.node_type} finished task execution. Exiting.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="SAPPHIRE EUCAIM Client Node")
    parser.add_argument("--node", required=True, help="Unique node identifier")
    parser.add_argument("--data", required=True, help="Path to input CSV file or directory")
    parser.add_argument("--model", default="mlp", help="Model architecture")
    parser.add_argument("--broker-host", default="localhost", help="RabbitMQ broker host or amqps:// URL")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = self_recognize_and_load(args.node, data_path=args.data)

    node = InteractiveNode(
        args.node,
        args.model,
        X_train,
        y_train,
        X_test,
        y_test,
        broker_host=args.broker_host
    )
    node.start()

if __name__ == "__main__":
    main()