import argparse
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.auth import bootstrap_admin

def main() -> None:
    parser = argparse.ArgumentParser(description="Platform administration commands")
    parser.add_argument("command", choices=["bootstrap-admin"])
    args = parser.parse_args()
    if args.command == "bootstrap-admin":
        with SessionLocal() as db:
            admin = bootstrap_admin(db, settings.initial_admin_email, settings.initial_admin_username, settings.initial_admin_password)
            print(f"Administrator is ready: {admin.username} ({admin.email})")

if __name__ == "__main__":
    main()
