from dotenv import load_dotenv
import os

load_dotenv()

addr = os.environ.get('GMAIL_ADDRESS', '')
pw = os.environ.get('GMAIL_APP_PASSWORD', '')

print(f"GMAIL_ADDRESS repr: {repr(addr)}")
print(f"GMAIL_APP_PASSWORD length: {len(pw)}  (should be exactly 16, no spaces)")
print(f"GMAIL_APP_PASSWORD repr (masked): {repr(pw[:2] + '*'*(len(pw)-4) + pw[-2:] if len(pw) > 4 else pw)}")
print(f"Has leading/trailing whitespace: {pw != pw.strip()}")
print(f"Contains a space: {' ' in pw}")
print(f"Contains a quote character: {chr(34) in pw or chr(39) in pw}")
