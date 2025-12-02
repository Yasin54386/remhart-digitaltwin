# REMHART Digital Twin - Deployment Reference Guide

**Complete Copy-Paste Deployment Commands for DigitalOcean**

This file contains the exact commands used for deployment. Keep this for future reference.

---

## Quick Deployment (Complete Copy-Paste)

### Prerequisites
- DigitalOcean Droplet: Ubuntu 22.04 LTS, 4GB RAM minimum
- Docker and Docker Compose installed
- Repository cloned

### One-Command Complete Deployment

```bash
# Complete automated deployment - Copy and paste this entire block
cat > .env << 'EOF'
MYSQL_ROOT_PASSWORD=RemhartRoot2024SecurePass!
MYSQL_DB=remhart_db
MYSQL_USER=remhart
MYSQL_PASSWORD=RemhartDb2024SecurePass!
SECRET_KEY=xK9pL2mN8vB4qR7sT1fG3hJ6dW0zY5cA9eU2iO4pQ7rT8sV1wX3yZ6bN5mK8jH4gF7wR3tY
FRONTEND_URL=http://$(curl -s ifconfig.me)
DEBUG=False
BACKEND_URL=http://backend:8001
DOMAIN=$(curl -s ifconfig.me)
EOF

echo "✓ Environment configured"

docker-compose up -d --build

echo "✓ Containers starting... waiting 60 seconds"
sleep 60

docker-compose ps

echo "Training ML models..."
docker-compose exec backend python setup_ml_models.py

echo "Seeding database..."
docker-compose exec backend python seed_database.py --points 5000 --scenario mixed

echo ""
echo "========================================="
echo "🎉 DEPLOYMENT COMPLETE!"
echo "========================================="
echo "Application URL: http://$(curl -s ifconfig.me)"
echo "========================================="
```

---

## Step-by-Step Deployment

### Step 1: Prepare Server

```bash
# Connect to droplet
ssh root@YOUR_DROPLET_IP

# Update system
apt-get update && apt-get upgrade -y

# Install Docker
apt-get install -y apt-transport-https ca-certificates curl software-properties-common git
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Verify
docker --version
docker-compose --version

# Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
```

### Step 2: Clone Repository

```bash
# Clean clone
cd ~
rm -rf remhart-digitaltwin
git clone https://github.com/Yasin54386/remhart-digitaltwin.git
cd remhart-digitaltwin

# Checkout main branch
git checkout main

# Merge deployment files (if needed)
git fetch origin
git merge origin/claude/deploy-digitalocean-01XX5Qd7hU1UgxhZoPUEibUB --no-edit

# Verify files exist
ls -la | grep -E "docker-compose|Dockerfile|QUICKSTART"
```

### Step 3: Configure Environment

```bash
# Auto-generate .env with secure defaults
cat > .env << 'EOF'
MYSQL_ROOT_PASSWORD=RemhartRoot2024SecurePass!
MYSQL_DB=remhart_db
MYSQL_USER=remhart
MYSQL_PASSWORD=RemhartDb2024SecurePass!
SECRET_KEY=xK9pL2mN8vB4qR7sT1fG3hJ6dW0zY5cA9eU2iO4pQ7rT8sV1wX3yZ6bN5mK8jH4gF7wR3tY
FRONTEND_URL=http://$(curl -s ifconfig.me)
DEBUG=False
BACKEND_URL=http://backend:8001
DOMAIN=$(curl -s ifconfig.me)
EOF

# Verify
cat .env
```

**To generate new passwords:**
```bash
# Generate secure password
openssl rand -base64 32

# Generate secret key
python3 -c "import secrets; print(secrets.token_urlsafe(50))"
```

### Step 4: Deploy Containers

```bash
# Build and start all services
docker-compose up -d --build

# Wait for containers to initialize
sleep 60

# Check status
docker-compose ps

# Expected output:
# NAME                STATUS
# remhart_backend     Up
# remhart_frontend    Up
# remhart_mysql       Up (healthy)
# remhart_nginx       Up

# View logs
docker-compose logs -f
# Press Ctrl+C to stop viewing
```

### Step 5: Initialize ML Models

```bash
# Access backend container
docker-compose exec backend bash

# Train all 16 ML models (~3-5 minutes)
python setup_ml_models.py

# Expected output:
# ======================================================================
#  REMHART DIGITAL TWIN - ML MODEL SETUP
# ======================================================================
# [STEP 1/2] Generating training data...
# ✓ Generated 10000 training samples
# [STEP 2/2] Training all ML models...
# ✓ Voltage Anomaly Detection
# ✓ Harmonic Distortion Detection
# ... (all 16 models)
# ✓ ML SETUP COMPLETE!

# Exit container (keep it open for next step)
```

### Step 6: Seed Database and Users

```bash
# Still inside backend container from Step 5

# 6a. Create default user accounts
python seed_users.py

# Expected output:
# ======================================================================
#  REMHART DIGITAL TWIN - USER MANAGEMENT
# ======================================================================
# Creating default users...
# ✓ Created user: admin        (Role: admin    ) Password: admin123
# ✓ Created user: operator     (Role: operator ) Password: operator123
# ✓ Created user: analyst      (Role: analyst  ) Password: analyst123
# ✓ Created user: viewer       (Role: viewer   ) Password: viewer123
# ✓ Created 4 new users
# ✓ USER SEEDING COMPLETE!

# 6b. Seed grid data (5000 data points, ~2-3 minutes)
python seed_database.py --points 5000 --scenario mixed

# Expected output:
# ================================================================================
#  REMHART DIGITAL TWIN - DATABASE SEEDING
# ================================================================================
# Generating 5000 data points (mixed scenario)...
# ✓ Generated 5000 data points
# Inserting data into database...
# ✓ Successfully inserted 5000 complete data records
# ✓ Data integrity verified - all counts match!
# ✓ DATABASE SEEDING COMPLETE!

# Exit container
exit
```

### Step 7: Verify Deployment

```bash
# Check containers
docker-compose ps

# Test application
curl http://localhost/health
# Should return: healthy

# Get your public URL
echo "Application URL: http://$(curl -s ifconfig.me)"

# Open in browser and verify:
# 1. Landing page loads at http://YOUR_IP/
# 2. Login page accessible at http://YOUR_IP/login/
# 3. Login with credentials:
#    Username: admin
#    Password: admin123
# 4. Dashboard loads after login
# 5. Charts display data
# 6. No errors in browser console
```

**Default Login Credentials:**
```
Username: admin      Password: admin123      Role: Full access
Username: operator   Password: operator123   Role: Control access
Username: analyst    Password: analyst123    Role: View + reports
Username: viewer     Password: viewer123     Role: Read-only

⚠️  IMPORTANT: Change these passwords in production!
```

---

## Database Seeding Options

```bash
# Access backend container first
docker-compose exec backend bash

# Quick test (1000 points = 50 minutes of data)
python seed_database.py

# Recommended for demo (5000 points = 4 hours of data)
python seed_database.py --points 5000 --scenario mixed

# Comprehensive (10000 points = 8 hours of data)
python seed_database.py --points 10000 --scenario normal

# Specific scenarios:
python seed_database.py --points 3000 --scenario normal          # Stable operation
python seed_database.py --points 3000 --scenario voltage_sag     # Voltage drops
python seed_database.py --points 3000 --scenario overcurrent     # Overload events
python seed_database.py --points 3000 --scenario frequency_drift # Frequency issues
python seed_database.py --points 5000 --scenario mixed          # Mixed anomalies (recommended)

# Clear and reseed
python seed_database.py --clear --points 2000

# View sample data
python seed_database.py --points 10 --sample

# Exit container
exit
```

**Scenario Descriptions:**
- `normal` - Stable grid operation, minimal anomalies
- `voltage_sag` - Includes voltage drop events (40-60% of data)
- `overcurrent` - Includes overload conditions (30-70% of data)
- `frequency_drift` - Includes frequency instability (20-80% of data)
- `mixed` - Mixed anomalies throughout dataset (every 10th point)

---

## Management Commands

### Container Management

```bash
# View status
docker-compose ps

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mysql
docker-compose logs -f nginx

# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v

# Start services
docker-compose up -d

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Container Access

```bash
# Access backend container
docker-compose exec backend bash

# Access frontend container
docker-compose exec frontend bash

# Access MySQL database
docker-compose exec mysql mysql -u root -p
# Enter MYSQL_ROOT_PASSWORD when prompted

# Run SQL query
docker-compose exec mysql mysql -u root -p -e "USE remhart_db; SELECT COUNT(*) FROM datetime_table;"
```

### Database Operations

```bash
# Backup database
docker-compose exec mysql mysqldump -u root -p remhart_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore database
docker-compose exec -T mysql mysql -u root -p remhart_db < backup_20250202_153045.sql

# Check database size
docker-compose exec mysql mysql -u root -p -e "SELECT table_schema 'Database', ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) 'Size (MB)' FROM information_schema.tables WHERE table_schema = 'remhart_db';"

# View table counts
docker-compose exec mysql mysql -u root -p remhart_db -e "
SELECT
  'datetime_table' as table_name, COUNT(*) as count FROM datetime_table
UNION ALL SELECT 'voltage_table', COUNT(*) FROM voltage_table
UNION ALL SELECT 'current_table', COUNT(*) FROM current_table
UNION ALL SELECT 'frequency_table', COUNT(*) FROM frequency_table
UNION ALL SELECT 'active_power_table', COUNT(*) FROM active_power_table
UNION ALL SELECT 'reactive_power_table', COUNT(*) FROM reactive_power_table;
"
```

### Resource Monitoring

```bash
# Container resource usage
docker stats

# Disk usage
df -h
docker system df

# Memory usage
free -h

# Check available ports
netstat -tuln | grep -E ':(80|443|8000|8001|3306)'
```

### Update Application

```bash
cd ~/remhart-digitaltwin

# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Check status
docker-compose ps
docker-compose logs -f
```

---

## Troubleshooting

### Issue: Containers won't start

```bash
# Check logs
docker-compose logs

# Check specific service
docker-compose logs backend

# Rebuild
docker-compose down
docker-compose up -d --build

# Check Docker service
systemctl status docker
systemctl restart docker
```

### Issue: Can't access application

```bash
# Check if containers are running
docker-compose ps

# Check firewall
ufw status

# Check nginx
docker-compose logs nginx

# Test locally
curl http://localhost/health

# Check if port is bound
netstat -tuln | grep :80
```

### Issue: Database connection errors

```bash
# Check MySQL health
docker-compose exec mysql mysqladmin ping -h localhost

# Check MySQL logs
docker-compose logs mysql

# Verify .env settings
cat .env

# Test connection from backend
docker-compose exec backend python -c "from app.database import check_db_connection; check_db_connection()"
```

### Issue: ML models not loading

```bash
# Check if models exist
docker-compose exec backend ls -la app/ml_models/trained/

# Retrain models
docker-compose exec backend python setup_ml_models.py

# Check backend logs
docker-compose logs backend | grep -i model
```

### Issue: Out of memory

```bash
# Check memory
free -h

# Add swap space
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Verify
free -h
swapon --show
```

### Issue: Port already in use

```bash
# Check what's using port 80
netstat -tuln | grep :80
lsof -i :80

# Kill process
fuser -k 80/tcp

# Or change port in docker-compose.yml
# ports:
#   - "8080:80"  # Change 80 to 8080
```

### Issue: Disk full

```bash
# Check disk space
df -h

# Clean Docker
docker system prune -a
docker volume prune

# Remove old images
docker images
docker rmi <image_id>

# Check log sizes
du -sh /var/lib/docker/containers/*/*-json.log
```

---

## Security Best Practices

### Change Default Passwords

```bash
# Edit .env file
nano .env

# Update these:
# MYSQL_ROOT_PASSWORD=<new-secure-password>
# MYSQL_PASSWORD=<new-secure-password>
# SECRET_KEY=<new-50-char-key>

# Restart containers
docker-compose restart
```

### Enable HTTPS/SSL

```bash
# Install Certbot
apt-get install -y certbot python3-certbot-nginx

# Stop nginx container
docker-compose stop nginx

# Obtain certificate
certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Create SSL directory
mkdir -p nginx/ssl

# Copy certificates
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/key.pem

# Edit nginx.conf and uncomment HTTPS server block
nano nginx/nginx.conf

# Restart nginx
docker-compose up -d nginx
```

### Automated Backups

```bash
# Create backup script
cat > /root/backup_remhart.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/root/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
cd /root/remhart-digitaltwin
docker-compose exec -T mysql mysqldump -u root -p${MYSQL_ROOT_PASSWORD} remhart_db > $BACKUP_DIR/remhart_db_$DATE.sql
find $BACKUP_DIR -name "remhart_db_*.sql" -mtime +7 -delete
echo "Backup completed: $DATE"
EOF

chmod +x /root/backup_remhart.sh

# Test backup
/root/backup_remhart.sh

# Schedule daily backups (2 AM)
crontab -e
# Add line:
0 2 * * * /root/backup_remhart.sh >> /var/log/remhart_backup.log 2>&1
```

---

## Performance Optimization

### Increase Connection Pool

Edit `backend/app/database.py`:
```python
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=20,        # Increase from 10
    max_overflow=40      # Increase from 20
)
```

### Enable Query Caching

Add Redis container to `docker-compose.yml`:
```yaml
redis:
  image: redis:alpine
  container_name: remhart_redis
  ports:
    - "6379:6379"
  networks:
    - remhart_network
```

### Optimize MySQL

```bash
# Edit MySQL configuration
docker-compose exec mysql bash
nano /etc/mysql/my.cnf

# Add:
# [mysqld]
# innodb_buffer_pool_size = 1G
# max_connections = 200
# query_cache_size = 64M

# Restart MySQL
docker-compose restart mysql
```

---

## Monitoring Setup

### Install Monitoring Tools

```bash
# Install netdata for system monitoring
bash <(curl -Ss https://my-netdata.io/kickstart.sh)

# Access: http://YOUR_IP:19999
```

### Application Health Checks

```bash
# Create health check script
cat > /root/health_check.sh << 'EOF'
#!/bin/bash
HEALTH_URL="http://localhost/health"
RESPONSE=$(curl -s $HEALTH_URL)

if [ "$RESPONSE" = "healthy" ]; then
    echo "$(date): Application is healthy"
else
    echo "$(date): Application health check failed!"
    docker-compose restart
fi
EOF

chmod +x /root/health_check.sh

# Schedule every 5 minutes
crontab -e
# Add:
*/5 * * * * /root/health_check.sh >> /var/log/remhart_health.log 2>&1
```

---

## Quick Reference

### Essential Commands

```bash
# Deploy
docker-compose up -d --build

# Stop
docker-compose down

# Restart
docker-compose restart

# Logs
docker-compose logs -f

# Status
docker-compose ps

# Update
git pull && docker-compose up -d --build

# Backup
docker-compose exec mysql mysqldump -u root -p remhart_db > backup.sql

# Access backend
docker-compose exec backend bash

# Seed users
docker-compose exec backend python seed_users.py

# Seed database
docker-compose exec backend python seed_database.py --points 5000 --scenario mixed

# List all users
docker-compose exec backend python seed_users.py --list

# Reset users (delete and recreate)
docker-compose exec backend python seed_users.py --reset
```

### Important Files

- `docker-compose.yml` - Container orchestration
- `.env` - Environment configuration (passwords, URLs)
- `backend/Dockerfile` - Backend container definition
- `frontend/Dockerfile` - Frontend container definition
- `nginx/nginx.conf` - Reverse proxy configuration
- `backend/seed_database.py` - Database seeding script
- `QUICKSTART.md` - Quick deployment guide
- `DEPLOYMENT.md` - Detailed deployment documentation

### Important Ports

- `80` - HTTP (nginx)
- `443` - HTTPS (nginx, if configured)
- `8000` - Frontend (Django) - internal only
- `8001` - Backend (FastAPI) - internal only
- `3306` - MySQL - internal only

### Default Credentials

**MySQL:**
- Username: `root` or `remhart`
- Password: Set in `.env` file
- Database: `remhart_db`

**Application:**
- Access via browser - no login required (add auth later if needed)

---

## Cost Estimation

**DigitalOcean Monthly Costs:**

| Configuration | Specs | Cost |
|--------------|-------|------|
| Basic Droplet | 2 vCPU, 4GB RAM | $24/mo |
| Production Droplet | 4 vCPU, 8GB RAM | $48/mo |
| Backups | 20% of droplet | +$4.80-9.60/mo |
| Load Balancer | Optional | $12/mo |
| Managed Database | Optional | $15/mo |

**Total:** $24-100/month depending on configuration

---

## Support Resources

- **QUICKSTART.md** - Fast deployment guide
- **DEPLOYMENT.md** - Detailed technical documentation
- **GitHub Repository** - https://github.com/Yasin54386/remhart-digitaltwin
- **Docker Documentation** - https://docs.docker.com
- **DigitalOcean Docs** - https://docs.digitalocean.com

---

## Deployment Checklist

### Pre-Deployment
- [ ] DigitalOcean droplet created (Ubuntu 22.04, 4GB RAM)
- [ ] SSH access configured
- [ ] Docker and Docker Compose installed
- [ ] Firewall configured (ports 22, 80, 443)

### Deployment
- [ ] Repository cloned
- [ ] `.env` file configured
- [ ] Containers built and started
- [ ] All containers show "Up" status
- [ ] ML models trained successfully
- [ ] Database seeded with grid data
- [ ] Application accessible via browser

### Post-Deployment
- [ ] Health checks passing
- [ ] Dashboard displays data correctly
- [ ] API endpoints responding
- [ ] Logs show no errors
- [ ] Automated backups configured
- [ ] Monitoring set up
- [ ] Documentation reviewed

---

## Version History

- **v1.0** - Initial deployment with Docker Compose
- **v1.1** - Added database seeding script
- **v1.2** - Added QUICKSTART guide and automated deployment

---

**Last Updated:** December 2, 2025
**Deployment Branch:** `claude/deploy-digitalocean-01XX5Qd7hU1UgxhZoPUEibUB`
**Production Branch:** `main`
