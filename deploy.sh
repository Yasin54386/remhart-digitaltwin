#!/bin/bash

# REMHART Digital Twin - Quick Deploy Script
# This script automates the deployment process on a fresh DigitalOcean Droplet

set -e

echo "======================================"
echo "REMHART Digital Twin Deployment"
echo "======================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and configure it first:"
    echo "  cp .env.example .env"
    echo "  nano .env"
    exit 1
fi

echo "Step 1: Installing Docker..."
if ! command -v docker &> /dev/null; then
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    echo "Docker installed successfully!"
else
    echo "Docker already installed."
fi

echo ""
echo "Step 2: Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed successfully!"
else
    echo "Docker Compose already installed."
fi

echo ""
echo "Step 3: Configuring firewall..."
ufw --force enable
ufw allow 22/tcp  # SSH
ufw allow 80/tcp  # HTTP
ufw allow 443/tcp # HTTPS
echo "Firewall configured!"

echo ""
echo "Step 4: Building and starting containers..."
docker-compose down 2>/dev/null || true
docker-compose up -d --build

echo ""
echo "Step 5: Waiting for services to be ready..."
sleep 10

echo ""
echo "Step 6: Checking container status..."
docker-compose ps

echo ""
echo "======================================"
echo "Deployment Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Train ML models by running:"
echo "   docker-compose exec backend python setup_ml_models.py"
echo ""
echo "2. Seed database with initial data:"
echo "   docker-compose exec backend python seed_database.py"
echo ""
echo "3. Access your application:"
echo "   http://$(curl -s ifconfig.me)"
echo ""
echo "4. View logs:"
echo "   docker-compose logs -f"
echo ""
echo "For more information, see DEPLOYMENT.md"
