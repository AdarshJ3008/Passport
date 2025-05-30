# utils/passport_ocr.py
import cv2, re, pytesseract, streamlit as st
from typing import Tuple, Optional, Dict

# Point to your local Tesseract binary
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ──────────────────────────────────────────────────────────────
def extract_viz_and_mrz(image_path: str) -> Tuple[Dict[str, str], Optional[str]]:
    # 1️⃣  VIZ OCR (whole page) --------------------------------
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    raw_viz   = pytesseract.image_to_string(bw)
    viz_lines = [ln.strip() for ln in raw_viz.splitlines() if ln.strip()]

    viz_data: Dict[str, str] = {}
    patterns = {
        "surname":         r"Surname\s*[:\-]?\s*(.+)",
        "given_names":     r"Given\s+Names?\s*[:\-]?\s*(.+)",
        "passport_number": r"Passport\s*No\.?\s*[:\-]?\s*([A-Z0-9]+)",
        "nationality":     r"Nationality\s*[:\-]?\s*(\w+)",
        "date_of_birth":   r"Date\s+of\s+Birth\s*[:\-]?\s*([\d\-\/]+)",
        "sex":             r"Sex\s*[:\-]?\s*([MF])",
        "place_of_birth":  r"Place\s+of\s+Birth\s*[:\-]?\s*(.+)",
        "date_of_issue":   r"Date\s+of\s+Issue\s*[:\-]?\s*([\d\-\/]+)",
        "date_of_expiry":  r"Date\s+of\s+Expiry\s*[:\-]?\s*([\d\-\/]+)",
        "authority":       r"Authority\s*[:\-]?\s*(.+)",
    }
    for ln in viz_lines:
        for key, pat in patterns.items():
            if key not in viz_data:
                m = re.search(pat, ln, re.IGNORECASE)
                if m:
                    viz_data[key] = m.group(1).strip()

    # 2️⃣  Crop bottom 20 % for MRZ ----------------------------
    h = gray.shape[0]
    mrz_crop = gray[int(h * 0.80):]
    mrz_crop = cv2.resize(mrz_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, mrz_bin = cv2.threshold(mrz_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    st.image(mrz_bin, caption="🧪 MRZ Cropped Region", use_column_width=True)

    # 3️⃣  Tesseract OCR (try OCR-B ➔ fall back to ENG) --------
    cfg_base = (
        "--oem 1 --psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< "
        "-c load_system_dawg=false -c load_freq_dawg=false "
    )
    def _ocr(lang: str) -> str:
        return pytesseract.image_to_string(mrz_bin, config=f"{cfg_base}-l {lang}")

    try:
        mrz_raw = _ocr("ocrb")
    except pytesseract.TesseractError:
        mrz_raw = _ocr("eng")

    st.subheader("📄 Full MRZ OCR Output (raw)")
    st.code(mrz_raw)

    # 4️⃣  Clean + fix ‘K’ glitch + collapse fillers + pad -----    
    def clean_line(s: str) -> str:
        """
        • strip spaces
        • replace spurious K beside fillers
        • collapse runs of filler chars to `<<`
        """
        s = s.strip().replace(" ", "")
        if not s:
            return s
        s = re.sub(r"K(?=<)", "<", s)          # 'K<'  ➜ '<<'
        s = re.sub(r"(?<=<)K", "<", s)         # '<K'  ➜ '<<'
        s = re.sub(r"K(<+)$", r"<\1", s)       # 'JOHNK<<<<' ➜ 'JOHN<<<<'
        s = re.sub(r"<<{2,}", "<<", s)         # '<<<'.. ➜ '<<'
        return s

    raw_lines = [clean_line(l) for l in mrz_raw.splitlines() if len(l.strip()) >= 25]

    def pad44(line: str) -> str:
        """
        • keep only A-Z, 0-9, '<'
        • if > 44 chars, keep right-most char (likely checksum) and
          trim fillers before it
        • if < 44, pad with '<'
        """
        line = re.sub(r"[^A-Z0-9<]", "", line.upper())
        if len(line) > 44:
            line = line[:43] + line[-1]          # preserve trailing checksum
        return line.ljust(44, "<")

    mrz_text: Optional[str] = None
    if len(raw_lines) >= 2:
        mrz_text = pad44(raw_lines[0]) + "\n" + pad44(raw_lines[1])

    st.subheader("✅ Cleaned & Padded MRZ (44 × 2)")
    st.code(mrz_text or "— could not build MRZ —")

    if not mrz_text:
        st.warning("⚠️ Unable to reconstruct a 2 × 44-char MRZ. Try a clearer photo.")

    return viz_data, mrz_text
