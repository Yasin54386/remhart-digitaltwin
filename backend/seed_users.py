#!/usr/bin/env python3
"""
REMHART Digital Twin - User Seeding Script
==========================================
Creates default user accounts for the system.

Usage:
    python seed_users.py

This creates 4 default users:
- admin (full access)
- operator (control access)
- analyst (view + reports)
- viewer (read-only)

Author: REMHART Team
Date: 2025
"""

import sys
from pathlib import Path
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.database import SessionLocal, Base, engine, check_db_connection
from app.models.db_models import User
from app.utils.security import hash_password


def create_default_users(db):
    """
    Create default user accounts.

    Creates 4 users with different roles:
    - admin: Full system access
    - operator: Control and monitoring access
    - analyst: View and report access
    - viewer: Read-only access
    """

    default_users = [
        {
            "username": "admin",
            "email": "admin@remhart.com",
            "password": "admin123",
            "full_name": "System Administrator",
            "role": "admin",
            "is_active": True
        },
        {
            "username": "operator",
            "email": "operator@remhart.com",
            "password": "operator123",
            "full_name": "Grid Operator",
            "role": "operator",
            "is_active": True
        },
        {
            "username": "analyst",
            "email": "analyst@remhart.com",
            "password": "analyst123",
            "full_name": "Data Analyst",
            "role": "analyst",
            "is_active": True
        },
        {
            "username": "viewer",
            "email": "viewer@remhart.com",
            "password": "viewer123",
            "full_name": "System Viewer",
            "role": "viewer",
            "is_active": True
        }
    ]

    print("\nCreating default users...")
    print("-" * 70)

    created_count = 0
    skipped_count = 0

    for user_data in default_users:
        # Check if user already exists
        existing_user = db.query(User).filter(User.username == user_data["username"]).first()

        if existing_user:
            print(f"  ⊗ User '{user_data['username']}' already exists - skipping")
            skipped_count += 1
            continue

        # Create new user
        hashed_password = hash_password(user_data["password"])

        new_user = User(
            username=user_data["username"],
            email=user_data["email"],
            hashed_password=hashed_password,
            full_name=user_data["full_name"],
            role=user_data["role"],
            is_active=user_data["is_active"],
            created_at=datetime.utcnow(),
            last_login=None
        )

        db.add(new_user)
        db.commit()

        print(f"  ✓ Created user: {user_data['username']:12} (Role: {user_data['role']:8}) Password: {user_data['password']}")
        created_count += 1

    print("-" * 70)
    print(f"✓ Created {created_count} new users")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} existing users")

    return created_count


def list_users(db):
    """List all users in the system."""
    print("\nCurrent Users:")
    print("-" * 70)
    print(f"{'Username':<15} {'Email':<25} {'Role':<10} {'Status':<10}")
    print("-" * 70)

    users = db.query(User).all()

    if not users:
        print("  No users found in database")
    else:
        for user in users:
            status = "Active" if user.is_active else "Inactive"
            print(f"{user.username:<15} {user.email:<25} {user.role:<10} {status:<10}")

    print("-" * 70)
    print(f"Total users: {len(users)}")


def delete_all_users(db):
    """Delete all users (use with caution!)"""
    user_count = db.query(User).count()

    if user_count == 0:
        print("No users to delete")
        return

    confirm = input(f"⚠️  Are you sure you want to delete all {user_count} users? (yes/no): ")

    if confirm.lower() == 'yes':
        db.query(User).delete()
        db.commit()
        print(f"✓ Deleted {user_count} users")
    else:
        print("Cancelled - no users deleted")


def main():
    """Main seeding function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage REMHART Digital Twin users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Create default users
  %(prog)s --list             # List all users
  %(prog)s --delete-all       # Delete all users (requires confirmation)
  %(prog)s --reset            # Delete all and recreate default users

Default Users Created:
  Username: admin      Password: admin123      Role: admin
  Username: operator   Password: operator123   Role: operator
  Username: analyst    Password: analyst123    Role: analyst
  Username: viewer     Password: viewer123     Role: viewer

⚠️  IMPORTANT: Change these default passwords in production!
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all existing users'
    )

    parser.add_argument(
        '--delete-all',
        action='store_true',
        help='Delete all users (requires confirmation)'
    )

    parser.add_argument(
        '--reset',
        action='store_true',
        help='Delete all users and recreate defaults'
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" REMHART DIGITAL TWIN - USER MANAGEMENT")
    print("=" * 70)

    # Check database connection
    if not check_db_connection():
        print("\n✗ Failed to connect to database!")
        print("Please check your .env file and ensure MySQL is running.")
        return 1

    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Create database session
    db = SessionLocal()

    try:
        if args.list:
            # List all users
            list_users(db)

        elif args.delete_all:
            # Delete all users
            delete_all_users(db)

        elif args.reset:
            # Reset: delete all and recreate
            print("\n🔄 Resetting users...")
            db.query(User).delete()
            db.commit()
            print("  ✓ Deleted all existing users")
            create_default_users(db)
            print("\n✓ Reset complete!")

        else:
            # Default: create users
            created = create_default_users(db)

            if created > 0:
                print("\n" + "=" * 70)
                print(" ✓ USER SEEDING COMPLETE!")
                print("=" * 70)
                print("\nDefault Login Credentials:")
                print("  Username: admin      Password: admin123")
                print("  Username: operator   Password: operator123")
                print("  Username: analyst    Password: analyst123")
                print("  Username: viewer     Password: viewer123")
                print("\n⚠️  IMPORTANT: Change these passwords in production!")
                print("\nYou can now login at: http://your-server-ip/login/")
            else:
                print("\n  All users already exist. Use --reset to recreate them.")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    exit(main())
