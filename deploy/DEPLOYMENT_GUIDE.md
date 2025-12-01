# REMHART Digital Twin - Deployment Guide

## Prerequisites
- Digital Ocean Droplet (Ubuntu 22.04 LTS, minimum 2GB RAM)
- IP Address: `159.89.199.144`
- Root access
- Email for SSL certificates: `yasin102875@gmail.com`

## Quick Deployment Steps

### 1. Connect to Your Droplet

```bash
ssh root@159.89.199.144
# Enter password: Remhart2024!Grid
```

### 2. Run Automated Deployment Script

```bash
# Download and run the deployment script
curl -fsSL https://raw.githubusercontent.com/YOUR_REPO/remhart-digitaltwin/main/deploy/deploy.sh | bash

# OR if you have the files locally:
cd /root
git clone YOUR_REPOSITORY_URL remhart-digitaltwin
cd remhart-digitaltwin/deploy
chmod +x deploy.sh
./deploy.sh
```

### 3. Upload Application Code

If not already cloned via git:

```bash
# From your local machine:
cd /path/to/remhart-digitaltwin
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '.git' \
  -e "ssh" . root@159.89.199.144:/opt/remhart-digitaltwin/
```

### 4. Start the Application

```bash
# On the droplet:
cd /opt/remhart-digitaltwin/deploy
docker-compose up -d
```

### 5. Check Status

```bash
# View logs
docker-compose logs -f

# Check containers
docker-compose ps

# View credentials
cat /opt/remhart-digitaltwin/CREDENTIALS.txt
```

## Manual Deployment (Alternative)

### Step 1: System Setup

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt-get install -y docker-compose-plugin
```

### Step 2: Prepare Application

```bash
# Create directory
mkdir -p /opt/remhart-digitaltwin
cd /opt/remhart-digitaltwin

# Upload your code here (via git clone or rsync)
```

### Step 3: Configure Environment

```bash
cd /opt/remhart-digitaltwin/deploy

# Generate secure passwords
MYSQL_ROOT_PASSWORD=$(openssl rand -base64 32)
MYSQL_PASSWORD=$(openssl rand -base64 32)
BACKEND_SECRET=$(openssl rand -base64 64)
FRONTEND_SECRET=$(openssl rand -base64 64)

# Create .env file
cat > .env << EOF
MYSQL_ROOT_PASSWORD=$MYSQL_ROOT_PASSWORD
MYSQL_DATABASE=remhart_db
MYSQL_USER=remhart_user
MYSQL_PASSWORD=$MYSQL_PASSWORD
EOF

# Create backend env
cat > .env.backend << EOF
DATABASE_URL=mysql+pymysql://remhart_user:$MYSQL_PASSWORD@mysql:3306/remhart_db
SECRET_KEY=$BACKEND_SECRET
API_URL=http://159.89.199.144
CORS_ORIGINS=["http://159.89.199.144"]
EOF

# Create frontend env
cat > .env.frontend << EOF
SECRET_KEY=$FRONTEND_SECRET
DEBUG=False
ALLOWED_HOSTS=159.89.199.144,localhost
API_URL=http://backend:8000
WS_URL=ws://159.89.199.144/ws
EOF
```

### Step 4: Configure Firewall

```bash
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

### Step 5: Launch Application

```bash
cd /opt/remhart-digitaltwin/deploy
docker-compose up -d
```

## Post-Deployment

### Initialize Database

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Create admin user (if needed)
docker-compose exec backend python setup_ml_models.py
```

### Access the Application

- **Frontend**: http://159.89.199.144
- **Backend API**: http://159.89.199.144/api
- **API Docs**: http://159.89.199.144/api/docs

### Setup SSL (Optional but Recommended)

If you have a domain name pointed to 159.89.199.144:

```bash
# Update nginx.conf with your domain
nano /opt/remhart-digitaltwin/deploy/nginx.conf

# Obtain SSL certificate
docker-compose run --rm certbot certonly --webroot \
  --webroot-path=/var/www/certbot \
  --email yasin102875@gmail.com \
  --agree-tos \
  --no-eff-email \
  -d yourdomain.com

# Uncomment HTTPS section in nginx.conf
nano /opt/remhart-digitaltwin/deploy/nginx.conf

# Restart nginx
docker-compose restart nginx
```

## Maintenance

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f nginx
```

### Restart Services

```bash
# All services
docker-compose restart

# Specific service
docker-compose restart backend
```

### Update Application

```bash
cd /opt/remhart-digitaltwin
git pull origin main
docker-compose down
docker-compose up -d --build
```

### Backup Database

```bash
# Backup
docker-compose exec mysql mysqldump -u root -p$MYSQL_ROOT_PASSWORD remhart_db > backup.sql

# Restore
docker-compose exec -T mysql mysql -u root -p$MYSQL_ROOT_PASSWORD remhart_db < backup.sql
```

## Troubleshooting

### Check Container Status

```bash
docker-compose ps
```

### View Container Logs

```bash
docker-compose logs --tail=100 backend
docker-compose logs --tail=100 frontend
```

### Restart Everything

```bash
docker-compose down
docker-compose up -d
```

### Check Database Connection

```bash
docker-compose exec backend python test_db_connection.py
```

### Access MySQL Console

```bash
docker-compose exec mysql mysql -u root -p
# Enter MYSQL_ROOT_PASSWORD from CREDENTIALS.txt
```

## Security Checklist

- [ ] Change default root password after deployment
- [ ] Set up SSL certificates if using a domain
- [ ] Configure firewall rules
- [ ] Regularly update system packages
- [ ] Set up automated backups
- [ ] Monitor logs for suspicious activity
- [ ] Keep Docker images updated

## Support

For issues, check:
1. Container logs: `docker-compose logs -f`
2. System logs: `journalctl -u docker`
3. Disk space: `df -h`
4. Memory usage: `free -h`

## Credentials

All generated credentials are saved in: `/opt/remhart-digitaltwin/CREDENTIALS.txt`

**IMPORTANT**: Save these credentials securely and delete the file after noting them down!
