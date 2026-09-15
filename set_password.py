"""
set_password.py — Run this locally to generate a new bcrypt password hash.
Paste the output into .streamlit/secrets.toml AND Streamlit Cloud Secrets.

Usage:
    python3 set_password.py
"""
import bcrypt, getpass, sys

def main():
    print("=" * 55)
    print("  Quant Terminal — Password Hash Generator")
    print("=" * 55)
    pw1 = getpass.getpass("Enter new password:    ")
    pw2 = getpass.getpass("Confirm new password:  ")

    if pw1 != pw2:
        print("\n❌  Passwords do not match. Exiting.")
        sys.exit(1)

    if len(pw1) < 8:
        print("\n⚠️  Warning: password is shorter than 8 characters.")

    hashed = bcrypt.hashpw(pw1.encode(), bcrypt.gensalt(rounds=12)).decode()

    print("\n✅  Hash generated successfully.\n")
    print("Paste this into .streamlit/secrets.toml AND Streamlit Cloud Secrets:")
    print(f'\npassword_hash = "{hashed}"\n')

if __name__ == "__main__":
    main()
