#!/usr/bin/env python3
"""
Database Connection Diagnostic Script
Tests MySQL connection and database setup
"""

import os
from dotenv import load_dotenv
import pymysql

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_SERVER = os.getenv("MYSQL_SERVER", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB = os.getenv("MYSQL_DB", "remhart_db")

print("="*60)
print("REMHART Database Connection Diagnostic")
print("="*60)
print(f"\nConfiguration:")
print(f"  Host: {MYSQL_SERVER}")
print(f"  Port: {MYSQL_PORT}")
print(f"  User: {MYSQL_USER}")
print(f"  Password: {'(set)' if MYSQL_PASSWORD else '(empty)'}")
print(f"  Database: {MYSQL_DB}")
print(f"  URL: {DATABASE_URL}")
print()

# Test 1: Check if MySQL is accessible
print("Test 1: Checking MySQL server connection...")
try:
    conn = pymysql.connect(
        host=MYSQL_SERVER,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    print("✓ Successfully connected to MySQL server")

    # Test 2: Check if database exists
    print("\nTest 2: Checking if database exists...")
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")
    databases = [db[0] for db in cursor.fetchall()]

    if MYSQL_DB in databases:
        print(f"✓ Database '{MYSQL_DB}' exists")
    else:
        print(f"✗ Database '{MYSQL_DB}' does NOT exist")
        print(f"\nAvailable databases: {', '.join(databases)}")
        print(f"\nTo create the database, run:")
        print(f"  mysql -u {MYSQL_USER} {'-p' if MYSQL_PASSWORD else ''} -e \"CREATE DATABASE {MYSQL_DB};\"")
        conn.close()
        exit(1)

    # Test 3: Check if we can connect to the specific database
    print("\nTest 3: Connecting to specific database...")
    conn.close()
    conn = pymysql.connect(
        host=MYSQL_SERVER,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    print(f"✓ Successfully connected to database '{MYSQL_DB}'")

    # Test 4: Check tables
    print("\nTest 4: Checking database tables...")
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]

    if tables:
        print(f"✓ Found {len(tables)} tables:")
        for table in tables:
            print(f"    - {table}")
    else:
        print("⚠ No tables found (database is empty)")
        print("  This is normal for a fresh installation.")
        print("  Tables will be created when you start the backend server.")

    conn.close()

    print("\n" + "="*60)
    print("✓ All diagnostic tests passed!")
    print("="*60)
    print("\nYour MySQL database is ready to use.")
    print("You can now start the backend server:")
    print("  cd /home/user/remhart-digitaltwin/backend")
    print("  python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload")

except pymysql.err.OperationalError as e:
    print(f"\n✗ MySQL connection failed!")
    print(f"Error: {e}")
    print("\nPossible solutions:")
    print("1. Make sure MySQL is running:")
    print("   sudo systemctl start mysql")
    print("\n2. Check your MySQL credentials:")
    print(f"   mysql -u {MYSQL_USER} {'-p' if MYSQL_PASSWORD else ''}")
    print("\n3. If password is required, update backend/.env:")
    print(f"   MYSQL_PASSWORD=your_password_here")
    print("\n4. Create the database:")
    print(f"   mysql -u {MYSQL_USER} {'-p' if MYSQL_PASSWORD else ''} -e \"CREATE DATABASE {MYSQL_DB};\"")
    exit(1)

except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
