import os
import streamlit as st
from PIL import Image

from utils.face_utils      import extract_face_embedding, match_face
from utils.passport_ocr    import extract_viz_and_mrz
from utils.mrz_decoder     import decode_mrz
# from utils.storage       import save_verified_user, is_user_verified  # disabled for test mode

# ─── Streamlit page config ─────────────────────────────────
st.set_page_config(page_title="Passport Verification", layout="centered")
st.title("🛂 Passport Verification System")

# ensure captured folder
os.makedirs("captured", exist_ok=True)

# ─── Step 1: Choose face input method ──────────────────────
face_source = st.radio(
    "How would you like to provide your face image?",
    ("Capture via camera", "Upload an image")
)

face_img_file = None
if face_source == "Capture via camera":
    cam = st.camera_input("📸 Capture your face")
    if cam:
        face_img_file = cam
else:
    up = st.file_uploader("📂 Upload your face image", type=["jpg", "jpeg", "png"])
    if up:
        face_img_file = up

if face_img_file:
    # Save the face image temporarily
    face_path = "captured/webcam_face.jpg"
    with open(face_path, "wb") as f:
        f.write(face_img_file.getbuffer())
    st.success("✅ Face image ready")
    st.image(face_img_file, caption="Provided Face", width=200)

    # Extract face embedding
    emb_face = extract_face_embedding(face_path)

    if emb_face is None:
        st.error("❌ No face detected in uploaded image.")
        st.stop()

    # ─── Step 2: Skip local verification for test mode ──────
    st.info("🆕 No previous user match checked (Test Mode – DB disabled).")

    # ─── Step 3: Upload passport ────────────────────────────
    passport_img = st.file_uploader("📄 Now upload your passport image", type=["jpg", "jpeg", "png"])
    if passport_img:
        passport_path = "captured/passport_photo.jpg"
        with open(passport_path, "wb") as f:
            f.write(passport_img.getbuffer())
        st.image(passport_img, caption="Uploaded Passport", use_column_width=True)

        # ─── Step 4: OCR & MRZ decoding ───────────────────────
        with st.spinner("🔍 Extracting VIZ + MRZ..."):
            viz_data, mrz_text = extract_viz_and_mrz(passport_path)
            decoded = decode_mrz(mrz_text) if mrz_text else {}

        st.json({"VIZ": viz_data, "MRZ Decoded": decoded})

        # ─── Step 5: Face matching against passport ───────────
        st.write("🔁 Matching your face against passport photo...")
        emb_passport = extract_face_embedding(passport_path)

        if emb_passport is not None:
            if match_face(emb_face, emb_passport):
                st.success("✅ Face matched! Verification successful.")
                st.caption("✔ No data has been saved. Test mode enabled.")
                # save_verified_user(emb_face, {**viz_data, **decoded})  # Disabled
            else:
                st.error("❌ Face does not match. Please try again.")
        else:
            st.error("❌ Could not detect a face in the passport photo.")

# ─── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption("🔒 Test Mode: No data is stored or saved.")
