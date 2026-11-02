import argparse
import secrets
import hashlib
import threading
import os
import random
import json
import time
import sys
import glob
import re
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from sapphire.broker import FLMessageBroker
from sapphire.fedproto import build_local_model, train_and_extract_prototypes

QUANTIZATION_SCALE = 100000
SECAGG_MODULUS = 2 ** 256
VALID_IMAGE_EXTS = ('.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff')


def generate_shares(secret: int, n: int, t: int, modulus: int) -> Dict[int, int]:
    # Generates Shamir Secret Sharing polynomial coefficients for secure aggregation
    coeffs = [secret] + [secrets.randbelow(modulus) for _ in range(t - 1)]
    return {
        i: sum(c * pow(i, e, modulus) for e, c in enumerate(coeffs)) % modulus
        for i in range(1, n + 1)
    }


def generate_aes_ctr_mask(seed_bytes: bytes, session_id: int, round_num: int, size: int) -> np.ndarray:
    # Generates a pseudo-random AES-CTR mask array for cryptographic dropout resilience
    nonce = hashlib.sha256(f"{session_id}:{round_num}".encode()).digest()[:16]
    cipher = Cipher(algorithms.AES(seed_bytes), modes.CTR(nonce))
    raw_bytes = cipher.encryptor().update(b'\x00' * (size * 32))

    uint64_chunks = np.frombuffer(raw_bytes, dtype=np.uint64).reshape(-1, 4)
    mask_ints = [
        (int(a) << 192) | (int(b) << 128) | (int(c) << 64) | int(d)
        for a, b, c, d in uint64_chunks
    ]
    return np.array(mask_ints, dtype=object) % SECAGG_MODULUS


def _win_long_path(path: str) -> str:
    # Adds the Windows extended-length prefix to prevent path length errors
    if os.name == 'nt':
        abs_p = os.path.abspath(path)
        if not abs_p.startswith('\\\\?\\'):
            return '\\\\?\\' + abs_p
    return path


def resolve_target_column(df: pd.DataFrame, labels_config: Dict[str, Any], local_target_override: str = None) -> str:
    # Dynamically identifies the classification target column from DataFrame metadata
    if local_target_override:
        if local_target_override in df.columns:
            print(f"🎯 Target column set via CLI override: '{local_target_override}'")
            return local_target_override
        raise ValueError(f"ERROR: Override target column '{local_target_override}' not found in CSV.")

    config_candidates = labels_config.get("target_columns") or [labels_config.get("target_column", "label")]
    label_mapping = labels_config.get("label_mapping", {})
    expected_classes = int(labels_config.get("num_classes", 2))

    name_matches = [c for c in config_candidates if c in df.columns]
    if len(name_matches) == 1:
        print(f"🎯 Target column resolved via name match: '{name_matches[0]}'")
        return name_matches[0]

    exclude_cols = {'split', 'id', 'patient id', 'patient_id', 'case_id', 'unnamed: 0', 'name', 'scan date'}
    pool = name_matches if len(name_matches) > 1 else [c for c in df.columns if str(c).lower().strip() not in exclude_cols]

    if label_mapping:
        norm_keys = {str(k).lower().strip() for k in label_mapping.keys()}
        mapping_matches = []
        for col in pool:  # Iterate through candidate columns to match mapping keys
            unique_vals = set(df[col].dropna().astype(str).str.lower().str.strip().unique())
            if unique_vals and unique_vals.issubset(norm_keys):
                mapping_matches.append(col)
        if len(mapping_matches) == 1:
            print(f"🎯 Target column resolved via label mapping match: '{mapping_matches[0]}'")
            return mapping_matches[0]
        elif len(mapping_matches) > 1:
            pool = mapping_matches

    cardinality_matches = [c for c in pool if df[c].dropna().nunique() == expected_classes]
    if len(cardinality_matches) == 1:
        print(f"🎯 Target column resolved via cardinality ({expected_classes} classes): '{cardinality_matches[0]}'")
        return cardinality_matches[0]
    elif len(cardinality_matches) > 1:
        chosen = cardinality_matches[-1]
        print(f"⚠️ Multiple columns matched {expected_classes} classes: {cardinality_matches}. Selecting: '{chosen}'.")
        return chosen

    if len(name_matches) > 1:
        return name_matches[0]

    raise ValueError(f"❌ Cannot identify target column automatically. Pool: {pool}. Columns: {list(df.columns)}")


def resolve_id_column(df: pd.DataFrame, dataset_config: Dict[str, Any] = None, local_id_override: str = None) -> str:
    # Locates the patient identifier column in the metadata CSV
    if local_id_override and local_id_override in df.columns:
        return local_id_override

    candidate_id_names = []
    if dataset_config and "patient_id_columns" in dataset_config:
        candidate_id_names.extend([str(c).lower().strip() for c in dataset_config["patient_id_columns"]])
    candidate_id_names.extend(['patientid', 'patient id', 'patient_id', 'proxid', 'case_id', 'case id', 'subject_id', 'id'])

    for candidate in candidate_id_names:  # Search possible ID names
        for col in df.columns:  # Check dataframe columns
            if str(col).lower().strip() == candidate:
                return col
    return df.columns[0]


def read_image_to_grayscale(file_path: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    # Loads and normalizes a DICOM or standard image slice into a 2D float array
    safe_path = _win_long_path(file_path)
    if file_path.lower().endswith('.dcm'):
        try:
            import pydicom
            dcm = pydicom.dcmread(safe_path, force=True)
            pixels = dcm.pixel_array.astype(np.float32)
            p_min, p_max = pixels.min(), pixels.max()
            if p_max > p_min:
                pixels = (pixels - p_min) / (p_max - p_min) * 255.0
            img = Image.fromarray(pixels.astype(np.uint8)).convert("L")
        except Exception as e:
            raise RuntimeError(f"Error reading DICOM {file_path}: {e}")
    else:
        img = Image.open(safe_path).convert("L")

    img = img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.expand_dims(arr, axis=0)


def extract_multimodal_features(row: pd.Series, img_tensor: np.ndarray, id_col: str, target_col: str,
                                patient_finding_count: int = 1) -> np.ndarray:
    # Extracts raw DICOM radiomic statistics and CSV spatial coordinates
    slice_2d = img_tensor[0]
    h, w = slice_2d.shape
    center_roi = slice_2d[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

    gy, gx = np.gradient(center_roi)
    grad_energy = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    std_val = float(np.std(slice_2d))
    skew_val = float(np.mean(((slice_2d - np.mean(slice_2d)) / (std_val + 1e-6)) ** 3))

    dcm_feats = [
        float(np.mean(slice_2d)),
        std_val,
        skew_val,
        float(np.percentile(slice_2d, 10)),
        float(np.percentile(slice_2d, 50)),
        float(np.percentile(slice_2d, 90)),
        float(np.mean(center_roi)),
        float(np.std(center_roi)),
        float(np.max(center_roi) - np.min(center_roi)),
        grad_energy
    ]

    pos_str = str(row.get('pos', '')) + " " + str(row.get('zone', ''))
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", pos_str)]
    while len(nums) < 3:  # Ensure 3 coordinate dimensions exist
        nums.append(0.0)
    x_coord, y_coord, z_coord = nums[:3]

    r_dist = float(np.sqrt(x_coord ** 2 + y_coord ** 2 + z_coord ** 2))
    xy_dist = float(np.sqrt(x_coord ** 2 + y_coord ** 2))

    zone_clean = re.sub(r"[^A-Za-z]", "", str(row.get('zone', ''))).upper()
    zone_onehot = [
        1.0 if zone_clean == 'PZ' else 0.0,
        1.0 if zone_clean == 'TZ' else 0.0,
        1.0 if zone_clean == 'AS' else 0.0,
        1.0 if zone_clean == 'SV' else 0.0
    ]
    fid_val = float(row.get('fid', 1.0)) if pd.notna(row.get('fid')) else 1.0

    extra_numeric = [
        float(row[col]) for col in row.index
        if col not in {id_col, target_col, 'fid', 'pos', 'zone', 'split'}
        and isinstance(row[col], (int, float, np.number)) and pd.notna(row[col])
    ]

    csv_feats = [fid_val, float(patient_finding_count), x_coord, y_coord, z_coord, r_dist, xy_dist] + zone_onehot + extra_numeric
    return np.array(csv_feats + dcm_feats, dtype=np.float32)


def _norm_folder_name(s: str) -> str:
    # Normalizes directory names for pattern matching
    return re.sub(r"[\s_]+", "-", str(s).lower().strip())


def _safe_split(X, y, test_size=0.2):
    # Splits dataset with stratified fallback if class distribution prevents stratification
    try:
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    except ValueError as e:
        print(f"⚠️ Stratified split failed ({e}). Falling back to uniform split.")
        return train_test_split(X, y, test_size=test_size, random_state=42)


def load_tabular_data(csv_path: str, labels_config: Dict[str, Any], local_target_override: str = None):
    # Loads and scales tabular datasets from CSV files
    print(f"📊 Loading tabular dataset: {csv_path}")
    df = pd.read_csv(_win_long_path(csv_path)).dropna(how='all')

    target_column = resolve_target_column(df, labels_config, local_target_override)
    label_mapping = labels_config.get("label_mapping", {})

    if label_mapping:
        norm_map = {str(k).lower().strip(): v for k, v in label_mapping.items()}
        df[target_column] = df[target_column].astype(str).str.lower().str.strip().map(norm_map)
        df = df.dropna(subset=[target_column])

    df[target_column] = df[target_column].astype(np.int64)

    if 'split' in df.columns:
        df_train = df[df['split'].astype(str).str.lower().str.strip() == 'train']
        df_test = df[df['split'].astype(str).str.lower().str.strip() == 'test']

        X_train_raw = df_train.drop(columns=[target_column, 'split'], errors='ignore').select_dtypes(include=[np.number]).values
        y_train = df_train[target_column].values
        X_test_raw = df_test.drop(columns=[target_column, 'split'], errors='ignore').select_dtypes(include=[np.number]).values
        y_test = df_test[target_column].values
    else:
        X = df.drop(columns=[target_column], errors='ignore').select_dtypes(include=[np.number]).values
        y = df[target_column].values
        X_train_raw, X_test_raw, y_train, y_test = _safe_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    return X_train, X_test, y_train, y_test


def load_image_data(data_path: str, labels_config: Dict[str, Any], dataset_config: Dict[str, Any] = None,
                    local_target_override: str = None, local_id_override: str = None):
    # Loads multimodal DICOM folders linked with metadata CSV manifests
    dataset_config = dataset_config or {}
    image_size = tuple(dataset_config.get("image_size", [128, 128]))
    label_mapping = labels_config.get("label_mapping", {})
    norm_label_mapping = {str(k).lower().strip(): v for k, v in label_mapping.items()}

    safe_data_path = _win_long_path(data_path)

    csv_candidates = []
    if os.path.isdir(safe_data_path):
        for root, _, files in os.walk(safe_data_path):  # Walk directory to locate CSV
            for f in files:
                if f.lower().endswith('.csv') and not f.startswith('.'):
                    csv_candidates.append(os.path.join(root, f))
        if not csv_candidates:
            parent_dir = os.path.dirname(safe_data_path)
            for f in os.listdir(parent_dir):  # Search parent directory as fallback
                if f.lower().endswith('.csv') and not f.startswith('.'):
                    csv_candidates.append(os.path.join(parent_dir, f))

    if csv_candidates:
        meta_csv = csv_candidates[0]
        print(f"📄 Found metadata file for Mode A linking: {meta_csv}")
        df = pd.read_csv(meta_csv).dropna(how='all')
        target_col = resolve_target_column(df, labels_config, local_target_override)
        id_col = resolve_id_column(df, dataset_config, local_id_override)

        if label_mapping:
            df[target_col] = df[target_col].astype(str).str.lower().str.strip().map(norm_label_mapping)
            df = df.dropna(subset=[target_col])
        df[target_col] = df[target_col].astype(np.int64)

        finding_counts = df[id_col].astype(str).str.strip().str.lower().value_counts().to_dict()
        patient_subdirs = {p.name.strip().lower(): p for p in [f for f in os.scandir(safe_data_path) if f.is_dir()]}

        X_multimodal_list, y_list = [], []
        n_loaded = 0

        for _, row in df.iterrows():  # Iterate through CSV manifest rows
            pat_id = str(row[id_col]).strip().lower()
            lbl = int(row[target_col])

            matched_files = []
            if pat_id in patient_subdirs:
                p_path = patient_subdirs[pat_id].path
                for root, _, files in os.walk(p_path):  # Collect DICOM files recursively
                    for f in files:
                        if f.lower().endswith(VALID_IMAGE_EXTS):
                            matched_files.append(os.path.join(root, f))
            else:
                pattern = os.path.join(safe_data_path, f"{pat_id}.*")
                matched_files = [f for f in glob.glob(pattern) if f.lower().endswith(VALID_IMAGE_EXTS)]

            if matched_files:
                matched_files.sort()
                fid_idx = int(row.get('fid', 1)) if pd.notna(row.get('fid')) else 1
                slice_idx = (len(matched_files) // 2 + (fid_idx - 1) * 3) % len(matched_files)
                target_slice = matched_files[slice_idx]

                img_tensor = read_image_to_grayscale(target_slice, image_size)
                pat_count = finding_counts.get(pat_id, 1)
                mm_features = extract_multimodal_features(row, img_tensor, id_col, target_col, pat_count)

                X_multimodal_list.append(mm_features)
                y_list.append(lbl)
                n_loaded += 1
                if n_loaded % 50 == 0:
                    print(f"   ... {n_loaded} patients linked", flush=True)

        if not X_multimodal_list:
            raise ValueError(f"❌ Could not match Patient IDs in column '{id_col}' to directory names in '{data_path}'.")

        X_raw = np.stack(X_multimodal_list)
        y = np.array(y_list, dtype=np.int64)

        X_train_raw, X_test_raw, y_train, y_test = _safe_split(X_raw, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)
        return X_train, X_test, y_train, y_test

    print("📁 No metadata CSV found. Inferring labels from folder structure (Mode B)...")
    folder_map = {_norm_folder_name(k): v for k, v in label_mapping.items()}
    keys_longest_first = sorted(folder_map.items(), key=lambda kv: -len(kv[0]))

    image_files = []
    for root, _, files in os.walk(safe_data_path):  # Discover images for Mode B fallback
        for f in files:
            if f.lower().endswith(VALID_IMAGE_EXTS):
                image_files.append(os.path.join(root, f))

    X_list, y_list = [], []
    for fpath in image_files:  # Process discovered image paths
        norm_path = fpath.lower().replace('\\', '/')
        parent_parts = [_norm_folder_name(p) for p in norm_path.split('/')[:-1]]

        matched_class = None
        for key, class_idx in folder_map.items():
            if key in parent_parts:
                matched_class = class_idx
                break
        if matched_class is None:
            for key, class_idx in keys_longest_first:
                if any(key in part for part in parent_parts):
                    matched_class = class_idx
                    break

        if matched_class is not None:
            X_list.append(read_image_to_grayscale(fpath, image_size))
            y_list.append(matched_class)

    if not X_list:
        raise ValueError("❌ No images matched any label_mapping classes from folder paths.")

    return _safe_split(np.stack(X_list), np.array(y_list, dtype=np.int64))


def universal_data_loader(node_type: str, data_path: str, labels_config: Dict[str, Any],
                          dataset_config: Dict[str, Any] = None, local_target_override: str = None,
                          local_id_override: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Dispatches loading routine based on whether path contains images or tabular data
    if data_path is None:
        data_path = os.environ.get("DATA_DIR", "./data")

    safe_path = _win_long_path(data_path)
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"❌ Cannot find dataset path at '{data_path}'")

    is_image = False
    if os.path.isdir(safe_path):
        for root, _, files in os.walk(safe_path):  # Check for presence of image files
            if any(f.lower().endswith(VALID_IMAGE_EXTS) for f in files):
                is_image = True
                break
    elif safe_path.lower().endswith(VALID_IMAGE_EXTS):
        is_image = True

    if is_image:
        return load_image_data(data_path, labels_config, dataset_config, local_target_override, local_id_override)
    else:
        csv_file = safe_path
        if os.path.isdir(safe_path):
            candidates = [os.path.join(safe_path, f) for f in os.listdir(safe_path) if f.endswith('.csv')]
            if not candidates:
                raise FileNotFoundError(f"ERROR: No CSV found in {data_path}")
            csv_file = candidates[0]
        return load_tabular_data(csv_file, labels_config, local_target_override)


def _predict(model, X_test, num_class, em_d, parameters, has_global_protos, device, batch_size=32):
    # Runs evaluation inference combining prototype cosine similarity and logits
    model.eval()
    gp_norm = None
    if has_global_protos and len(parameters) > 0:
        gp_tensor = torch.tensor(parameters[0], dtype=torch.float32).reshape(num_class, em_d).to(device)
        gp_norm = F.normalize(gp_tensor, dim=1)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):  # Batch inference loop
            xb = torch.as_tensor(X_test[i:i + batch_size], dtype=torch.float32).to(device)
            v, logits = model(xb)
            if gp_norm is not None:
                similarity = torch.mm(F.normalize(v, dim=1), gp_norm.T)
                cos_probs = F.softmax(similarity / 0.5, dim=1)
                logit_probs = F.softmax(logits, dim=1)
                probs = 0.5 * cos_probs + 0.5 * logit_probs
            else:
                probs = logits
            preds.append(torch.argmax(probs, dim=1).cpu().numpy())
    return np.concatenate(preds)


class InteractiveNode:
    def __init__(self, node_type, requested_model, data_path, broker_host="localhost",
                 local_target_override=None, local_id_override=None):
        # Initializes the client node parameters and communication state
        self.broker = FLMessageBroker(host=broker_host)
        self.node_type = node_type
        self.requested_model = requested_model
        self.data_path = data_path
        self.local_target_override = local_target_override
        self.local_id_override = local_id_override

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.start_event = threading.Event()
        self.finish_event = threading.Event()
        self.my_id = -1
        self.registered = False
        self.seeds = {}
        self.sess_id = -1
        self._train_lock = threading.Lock()

    def handle_messages(self, msg):
        # Handles incoming broker messages according to training protocol state
        cmd = msg.get("command") or msg.get("type")

        if cmd == "SETUP":
            incoming_session = msg.get("session_id")
            locked_session = getattr(self, "setup_session_id", None)
            if locked_session is None:
                self.setup_session_id = incoming_session
            elif incoming_session != locked_session:
                return

            if not self.registered:
                if getattr(self, 'data_loaded', False):
                    self.broker.send_to_server({
                        "type": "REGISTER_REPLY",
                        "node_name": self.node_type
                    })
                    return

                self.data_loaded = True
                self.num_class = int(msg["num_classes"])
                self.model_groups = msg.get("model_groups", {})
                if not self.requested_model:
                    self.requested_model = msg.get("global_model", "")

                labels_config = msg.get("labels_config", {})
                dataset_config = msg.get("dataset_config", {})
                print(f" Loading dataset dynamically for {self.node_type}...")
                try:
                    self.X_train, self.X_test, self.y_train, self.y_test = universal_data_loader(
                        self.node_type,
                        self.data_path,
                        labels_config,
                        dataset_config,
                        self.local_target_override,
                        self.local_id_override
                    )

                    print(" ⚖️ Balancing training data...")
                    rng_balance = np.random.RandomState(42)
                    class_counts = np.bincount(self.y_train, minlength=self.num_class)
                    max_count = np.max(class_counts)

                    X_balanced, y_balanced = [], []
                    for c in range(self.num_class):  # Oversample each class to max size
                        idx_c = np.where(self.y_train == c)[0]
                        if len(idx_c) > 0:
                            idx_resampled = rng_balance.choice(idx_c, size=max_count, replace=True)
                            X_balanced.append(self.X_train[idx_resampled])
                            y_balanced.append(self.y_train[idx_resampled])

                    if X_balanced:
                        self.X_train = np.vstack(X_balanced)
                        self.y_train = np.concatenate(y_balanced)

                        shuff_idx = rng_balance.permutation(len(self.y_train))
                        self.X_train = self.X_train[shuff_idx]
                        self.y_train = self.y_train[shuff_idx]

                except Exception as e:
                    print(f"❌ FATAL LOAD ERROR: {e}")
                    sys.stdout.flush()
                    os._exit(1)

                print(f" {self.node_type}: train per class={np.bincount(self.y_train, minlength=self.num_class).tolist()}, "
                      f"test per class={np.bincount(self.y_test, minlength=self.num_class).tolist()}, "
                      f"train shape={self.X_train.shape}", flush=True)

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
            for peer_id_str, peer_comm_pub_str in self.peer_comm_keys.items():  # Encrypt shares for peers
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

            for sender_id_str, enc_payload in my_inbound_shares.items():  # Decrypt incoming shares
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

            for o_id, pk in self.peer_keys.items():  # Derive pairwise masking seeds
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
            survivors_raw = msg.get("survivors", [])
            dropped_raw = msg.get("dropped", [])

            reply_shares = {"loc": {}, "priv": {}}
            for s in survivors_raw:  # Gather local seed shares for survivors
                if str(s) in self.stored_shares and "loc" in self.stored_shares[str(s)]:
                    reply_shares["loc"][str(s)] = self.stored_shares[str(s)]["loc"][rnd - 1]

            for d in dropped_raw:  # Gather private key shares for dropped clients
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
        # Executes local training epoch, extracts prototypes, and transmits masked updates
        with self._train_lock:
            while getattr(self, 'model', None) is None or getattr(self, 'optimizer', None) is None:
                time.sleep(0.1)

            rnd = msg["round"]
            global_protos_list = msg.get("global_prototypes", [])
            has_global_protos = len(global_protos_list) > 0
            parameters = [np.array(global_protos_list, dtype=np.float32)] if has_global_protos else []
            device = next(self.model.parameters()).device

            current_lr = 0.001 * (0.95 ** (rnd - 1))
            for pg in self.optimizer.param_groups:
                pg['lr'] = current_lr

            print(f" {self.node_type}: round {rnd} local training started "
                  f"({len(self.y_train)} train / {len(self.y_test)} test samples)...", flush=True)
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
            acc = float(accuracy_score(self.y_test, preds))
            balanced_acc = float(balanced_accuracy_score(self.y_test, preds))
            print(f" {self.node_type}: round {rnd} local acc={acc:.3f} balanced={balanced_acc:.3f}. "
                  f"Sending masked update...", flush=True)

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
            mask_size = len(scaled_v_i)

            p_mask = np.zeros(mask_size, dtype=object)
            for o_id, seed in self.seeds.items():  # Accumulate pairwise cryptographic masks
                mask = generate_aes_ctr_mask(seed, self.sess_id, rnd, mask_size)
                p_mask = (p_mask + mask) % SECAGG_MODULUS if self.my_id > int(o_id) \
                    else (p_mask - mask) % SECAGG_MODULUS

            current_seed = self.local_seeds[rnd - 1]
            current_seed_bytes = current_seed.to_bytes(32, 'big')

            combined_mask = (p_mask + generate_aes_ctr_mask(current_seed_bytes,
                                                             self.sess_id, rnd, mask_size)) % SECAGG_MODULUS
            y_i_str = [str(x) for x in (scaled_v_i + combined_mask) % SECAGG_MODULUS]

            self.broker.send_to_server({
                "command": "UPDATE",
                "client_id": self.my_id,
                "round": rnd,
                "masked_data": y_i_str
            })

    def start(self):
        # Starts message listener thread, builds local model, and runs main loop
        print(f" SAPPHIRE Node starting: {self.node_type}")
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

        self.model, _ = build_local_model(self.group, self.model_name,
                                          self.X_train, self.y_train,
                                          self.em_d, self.num_class)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

        self.finish_event.wait(timeout=3600)
        print(f"Node {self.node_type} finished task execution. Exiting.")
        sys.exit(0)


def main():
    # Parses command line arguments and initializes client node
    parser = argparse.ArgumentParser(description="SAPPHIRE Universal Federated Client")
    parser.add_argument("--node", required=True, help="Unique node identifier")
    parser.add_argument("--data", required=True, help="Path to input data directory or file")
    parser.add_argument("--model", default=None, help="Model architecture")
    parser.add_argument("--broker-host", default=None, help="RabbitMQ broker host or amqps:// URL")
    parser.add_argument("--target-col", default=None, help="Override target column name")
    parser.add_argument("--patient-id-col", default=None, help="Override patient ID column name")
    args = parser.parse_args()

    broker_url = args.broker_host or os.environ.get("RABBITMQ_URL") or "localhost"

    node = InteractiveNode(
        args.node,
        args.model,
        args.data,
        broker_host=broker_url,
        local_target_override=args.target_col,
        local_id_override=args.patient_id_col
    )
    node.start()


if __name__ == "__main__":
    main()