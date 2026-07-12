"""
One-time helper: delete the IAUI EDGAR cache file so the next request
is guaranteed to trigger a fresh fetch. Run this directly:

    python clear_iaui_cache.py

Prints exactly what it found and did, so there's no ambiguity.
"""
import os

cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edgar_cache', 'IAUI.json')

print(f"Looking for: {cache_path}")
if os.path.exists(cache_path):
    print(f"FOUND IT. File size: {os.path.getsize(cache_path)} bytes")
    os.remove(cache_path)
    if os.path.exists(cache_path):
        print("❌ DELETE FAILED — file still exists after os.remove()! Check file permissions.")
    else:
        print("✅ DELETED successfully. Confirmed gone.")
else:
    print("File does not exist — nothing to delete (this is also a valid/clean state).")
