#!/bin/bash

# REMHART Digital Twin Deployment Script
# Run this script on your Digital Ocean droplet as root

set -e

echo "================================"
echo "REMHART Digital Twin Deployment"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${YELLOW}→ $1${NC}"; }

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root (use sudo)"
    exit 1
fi

print_info "Starting deployment process..."

# Update system
print_info "Updating system packages..."
apt-get update
apt-get upgrade -y
print_success "System updated"

# Install Docker
print_info "Installing Docker..."
if ! command -v docker &> /dev/null; then
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    systemctl start docker
    systemctl enable docker
    print_success "Docker installed"
else
    print_success "Docker already installed"
fi

# Install Docker Compose
print_info "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    print_success "Docker Compose installed"
else
    print_success "Docker Compose already installed"
fi

# Install Git
print_info "Installing Git..."
if ! command -v git &> /dev/null; then
    apt-get install -y git
    print_success "Git installed"
else
    print_success "Git already installed"
fi

# Create application directory
print_info "Creating application directory..."
APP_DIR="/opt/remhart-digitaltwin"
mkdir -p $APP_DIR
cd $APP_DIR
print_success "Application directory created at $APP_DIR"

# Generate secure passwords
print_info "Generating secure passwords..."
MYSQL_ROOT_PASSWORD=$(openssl rand -base64 32)
MYSQL_PASSWORD=$(openssl rand -base64 32)
BACKEND_SECRET=$(openssl rand -base64 64)
FRONTEND_SECRET=$(openssl rand -base64 64)

# Create environment files
print_info "Creating environment configuration..."

# Main .env for docker-compose
cat > $APP_DIR/deploy/.env << EOF
MYSQL_ROOT_PASSWORD=$MYSQL_ROOT_PASSWORD
MYSQL_DATABASE=remhart_db
MYSQL_USER=remhart_user
MYSQL_PASSWORD=$MYSQL_PASSWORD
EOF

# Backend environment
cat > $APP_DIR/deploy/.env.backend << EOF
DATABASE_URL=mysql+pymysql://remhart_user:$MYSQL_PASSWORD@mysql:3306/remhart_db
SECRET_KEY=$BACKEND_SECRET
API_URL=http://159.89.199.144
CORS_ORIGINS=["http://159.89.199.144", "http://localhost:8001"]
EOF

# Frontend environment
cat > $APP_DIR/deploy/.env.frontend << EOF
SECRET_KEY=$FRONTEND_SECRET
DEBUG=False
ALLOWED_HOSTS=159.89.199.144,localhost,127.0.0.1
API_URL=http://backend:8000
WS_URL=ws://159.89.199.144/ws
EOF

print_success "Environment files created"

# Set up firewall
print_info "Configuring firewall..."
ufw --force enable
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
print_success "Firewall configured"

# Create certbot directories
print_info "Creating SSL certificate directories..."
mkdir -p $APP_DIR/deploy/certbot/conf
mkdir -p $APP_DIR/deploy/certbot/www
print_success "Certificate directories created"

# Save credentials to file
print_info "Saving credentials..."
CREDS_FILE="$APP_DIR/CREDENTIALS.txt"
cat > $CREDS_FILE << EOF
========================================
REMHART DIGITAL TWIN - ACCESS CREDENTIALS
========================================
Generated: $(date)

MySQL Database:
  Root Password: $MYSQL_ROOT_PASSWORD
  Database: remhart_db
  User: remhart_user
  Password: $MYSQL_PASSWORD

Backend API:
  Secret Key: $BACKEND_SECRET

Frontend:
  Secret Key: $FRONTEND_SECRET

Access URLs:
  Frontend: http://159.89.199.144
  Backend API: http://159.89.199.144/api
  WebSocket: ws://159.89.199.144/ws

IMPORTANT:
- Keep this file secure
- Delete after saving credentials elsewhere
- Change default admin passwords after first login
========================================
EOF

chmod 600 $CREDS_FILE
print_success "Credentials saved to $CREDS_FILE"

# Display final instructions
echo ""
echo "================================"
print_success "Deployment preparation complete!"
echo "================================"
echo ""
print_info "Next steps:"
echo "1. Upload your code to $APP_DIR"
echo "2. Run: cd $APP_DIR/deploy && docker-compose up -d"
echo "3. Check logs: docker-compose logs -f"
echo "4. View credentials: cat $CREDS_FILE"
echo ""
print_info "Your credentials have been saved to: $CREDS_FILE"
echo ""
