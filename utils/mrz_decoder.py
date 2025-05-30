import re
from typing import Dict

# Mapping for MRZ characters to numeric values
_CHAR_VALUES = {str(i): i for i in range(10)}
_CHAR_VALUES.update({chr(i + ord('A')): 10 + i for i in range(26)})
_CHAR_VALUES['<'] = 0

# Weights cycle [7, 3, 1]
_WEIGHTS = [7, 3, 1]

def _checksum(field: str) -> int:
    """
    Compute the ICAO 9303 checksum for a given field.
    """
    total = 0
    for i, ch in enumerate(field):
        val = _CHAR_VALUES.get(ch, 0)
        weight = _WEIGHTS[i % 3]
        total += val * weight
    return total % 10

def safe_digit(ch: str) -> int:
    """Convert a character to int safely. Return -1 if not a digit."""
    return int(ch) if ch.isdigit() else -1

def decode_mrz(mrz: str) -> Dict[str, object]:
    """
    Decode a two-line MRZ (TD3 format) into its constituent fields,
    along with checksum validation flags.
    """
    lines = mrz.splitlines()
    if len(lines) != 2 or any(len(ln) != 44 for ln in lines):
        raise ValueError("MRZ must be two lines of length 44 each")

    l1, l2 = lines

    # Line 1
    doc_type = l1[0]
    issuing_country = l1[2:5]
    name_field = l1[5:]
    parts = name_field.rstrip('<').split('<<', 1)
    surname = parts[0].replace('<', ' ')
    given_names = parts[1].replace('<', ' ') if len(parts) > 1 else ''

    # Line 2
    passport_number = l2[0:9]
    passport_ck = safe_digit(l2[9])
    nationality = l2[10:13]
    dob = l2[13:19]
    dob_ck = safe_digit(l2[19])
    sex = l2[20]
    expiry = l2[21:27]
    expiry_ck = safe_digit(l2[27])
    personal_number = l2[28:42]
    personal_ck = safe_digit(l2[42])
    final_ck = safe_digit(l2[43])

    # Validate checksums (if safe)
    passport_valid = passport_ck != -1 and (_checksum(passport_number) == passport_ck)
    dob_valid = dob_ck != -1 and (_checksum(dob) == dob_ck)
    expiry_valid = expiry_ck != -1 and (_checksum(expiry) == expiry_ck)
    personal_valid = personal_ck != -1 and (_checksum(personal_number) == personal_ck)

    # Composite field for final checksum
    composite = (
        passport_number + l2[9] +
        dob + l2[19] +
        sex +
        expiry + l2[27] +
        personal_number + l2[42]
    )
    final_valid = final_ck != -1 and (_checksum(composite) == final_ck)

    return {
        "document_type": doc_type,
        "issuing_country": issuing_country,
        "surname": surname.strip(),
        "given_names": given_names.strip(),
        "passport_number": passport_number,
        "passport_number_valid": passport_valid,
        "nationality": nationality,
        "date_of_birth": dob,
        "date_of_birth_valid": dob_valid,
        "sex": sex,
        "date_of_expiry": expiry,
        "date_of_expiry_valid": expiry_valid,
        "personal_number": personal_number.rstrip('<'),
        "personal_number_valid": personal_valid,
        "final_checksum_valid": final_valid
    }
