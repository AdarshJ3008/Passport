import numpy as np
import cv2
from insightface.app import FaceAnalysis

# Initialize once
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

def extract_face_embedding(image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path)
    faces = face_analyzer.get(img)

    if not faces:
        return None

    return faces[0].embedding  # 512-d embedding


def match_face(
    emb1: np.ndarray,
    emb2: np.ndarray,
    threshold: float = 0.6,
    metric: str = "cosine"
) -> bool:
    if metric == "euclidean":
        dist = np.linalg.norm(emb1 - emb2)
        return dist <= threshold

    # Cosine similarity
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    cosine_sim = np.dot(emb1_norm, emb2_norm)
    return (1.0 - cosine_sim) <= threshold
