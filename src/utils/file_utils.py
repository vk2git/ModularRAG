import hashlib

def calculate_file_hash(filepath):
    """
    Generate a SHA-256 hash of file content
    """

    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None