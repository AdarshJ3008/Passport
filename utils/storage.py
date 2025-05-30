import os
import json
import base64
import numpy as np
from typing import Tuple, Optional, Dict
from cryptography.fernet import Fernet

DB_PATH = "data/verified_users.json"
KEY_PATH = "data/encryption.key"


# ─── Encryption Setup ─────────────────────────────
def _load_encryption_key() -> bytes:
    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)
    else:
        with open(KEY_PATH, "rb") as f:
            key = f.read()
    return key

fernet = Fernet(_load_encryption_key())


# ─── DB Setup ──────────────────────────────────────
def _ensure_db():
    dirname = os.path.dirname(DB_PATH)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, "w") as f:
            json.dump({}, f, indent=4)


# ─── Save Verified User ────────────────────────────
def save_verified_user(embedding: np.ndarray, metadata: dict):
    """
    Save user in structured format:
    {
      "user_1": {
         "embedding": <encrypted>,
         "metadata": <encrypted>,
         "name":     <encrypted>
      }
    }
    """
    _ensure_db()
    with open(DB_PATH, "r") as f:
        users = json.load(f)

    uid = f"user_{len(users) + 1}"

    # Convert embedding to bytes → encrypt
    emb_bytes = embedding.astype(np.float64).tobytes()
    emb_enc = fernet.encrypt(base64.b64encode(emb_bytes)).decode()

    # Encrypt metadata (including name field if present)
    metadata_json = json.dumps(metadata)
    metadata_enc = fernet.encrypt(metadata_json.encode()).decode()

    name_enc = fernet.encrypt(metadata.get("given_names", "Unknown").encode()).decode()

    users[uid] = {
        "name": name_enc,
        "embedding": emb_enc,
        "metadata": metadata_enc
    }

    with open(DB_PATH, "w") as f:
        json.dump(users, f, indent=4)


# ─── Load Verified Users ───────────────────────────
def load_verified_users() -> Dict[str, dict]:
    """
    Return decrypted dict:
    {
      "user_1": {
         "embedding": <np.ndarray>,
         "metadata": {dict},
         "name": "John"
      }
    }
    """
    _ensure_db()
    with open(DB_PATH, "r") as f:
        users = json.load(f)

    decrypted_users = {}

    for uid, rec in users.items():
        try:
            # Decrypt embedding
            emb_enc = rec["embedding"]
            emb_dec = base64.b64decode(fernet.decrypt(emb_enc.encode()))
            emb_array = np.frombuffer(emb_dec, dtype=np.float64)

            # Decrypt metadata
            metadata_enc = rec["metadata"]
            metadata = json.loads(fernet.decrypt(metadata_enc.encode()).decode())

            # Decrypt name
            name_enc = rec.get("name", "")
            name = fernet.decrypt(name_enc.encode()).decode() if name_enc else "Unknown"

            decrypted_users[uid] = {
                "embedding": emb_array,
                "metadata": metadata,
                "name": name
            }

        except Exception as e:
            print(f"⚠️ Skipped corrupted user {uid}: {e}")
            continue

    return decrypted_users


# ─── Check If Verified ─────────────────────────────
def is_user_verified(embedding: np.ndarray, threshold: float = 0.6) -> Tuple[bool, Optional[Dict]]:
    users = load_verified_users()
    for user_id, user in users.items():
        dist = np.linalg.norm(embedding - user["embedding"])
        if dist <= threshold:
            return True, user["metadata"]
    return False, None
