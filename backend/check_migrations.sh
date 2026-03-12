#!/bin/bash
# check_migrations.sh — verify Alembic migrations are up to date and apply if not

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=== Alembic migration status ==="
CURRENT=$(alembic current 2>&1)
echo "$CURRENT"

# Check if there are pending migrations (heads not in current)
PENDING=$(alembic heads 2>&1)
echo ""
echo "=== Latest head(s) ==="
echo "$PENDING"

# Compare: if current output contains "(head)", we're up to date
if echo "$CURRENT" | grep -q "(head)"; then
    echo ""
    echo "[OK] Database is up to date — no migrations needed."
else
    echo ""
    echo "[PENDING] Unapplied migrations detected. Running 'alembic upgrade head'..."
    alembic upgrade head
    echo ""
    echo "[DONE] Migrations applied successfully."
    echo ""
    echo "=== New status ==="
    alembic current
fi
