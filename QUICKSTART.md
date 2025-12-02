# REMHART Digital Twin - Complete Deployment Guide

**From Zero to Running in 15 Minutes**

This guide takes you from a fresh DigitalOcean droplet to a fully functional REMHART Digital Twin with ML models and seeded data.

---

## Prerequisites

- DigitalOcean account
- SSH client
- 15 minutes of time

---

## Step 1: Create DigitalOcean Droplet (5 minutes)

### 1.1 Create Droplet

1. Log in to [DigitalOcean](https://cloud.digitalocean.com)
2. Click **"Create"** → **"Droplets"**
3. Configure:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: Basic
   - **CPU**: Regular (2 vCPUs, 4GB RAM) - **Minimum recommended**
   - **Datacenter**: Choose closest to your location
   - **Authentication**: SSH key (recommended) or Password
   - **Hostname**: `remhart-digitaltwin`
4. Click **"Create Droplet"**
5. Wait for droplet to be created (~60 seconds)
6. **Copy your Droplet IP** (e.g., `159.89.199.144`)

---

## Step 2: Connect to Your Droplet

```bash
ssh root@YOUR_DROPLET_IP
```

Replace `YOUR_DROPLET_IP` with your actual IP address.

**Example:**
```bash
ssh root@159.89.199.144
```

---

## Step 3: Install Docker & Docker Compose (2 minutes)

Run these commands on your droplet:

```bash
# Update package list
apt-get update

# Install required packages
apt-get install -y apt-transport-https ca-certificates curl software-properties-common git

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

# Add Docker repository
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Install Docker
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

You should see Docker and Docker Compose versions displayed.

---

## Step 4: Clone Repository (1 minute)

```bash
# Clone the repository
git clone https://github.com/Yasin54386/remhart-digitaltwin.git

# Enter the directory
cd remhart-digitaltwin

# Checkout the deployment branch (or use main)
git checkout main
```

---

## Step 5: Configure Environment (2 minutes)

```bash
# Copy example environment file
cp .env.example .env

# Edit the environment file
nano .env
```

### Update these values in `.env`:

**Required Changes:**
```bash
# Generate a secure root password
MYSQL_ROOT_PASSWORD=your-secure-root-password-here

# Generate a secure database password
MYSQL_PASSWORD=your-secure-db-password-here

# Generate a secure secret key (run the command below to generate)
SECRET_KEY=your-50-character-secret-key-here

# Update with your droplet IP
FRONTEND_URL=http://YOUR_DROPLET_IP
```

### Generate Secure Secret Key:

```bash
# Generate a 50-character secret key
python3 -c "import secrets; print(secrets.token_urlsafe(50))"
```

Copy the output and paste it as your `SECRET_KEY` value.

### Example `.env` file:

```bash
MYSQL_ROOT_PASSWORD=MySecureRootPass123!
MYSQL_DB=remhart_db
MYSQL_USER=remhart
MYSQL_PASSWORD=MySecureDbPass456!

SECRET_KEY=xK9pL2mN8vB4qR7sT1fG3hJ6dW0zY5cA9eU2iO4pQ7rT8sV1wX3yZ6bN5mK8jH4gF
FRONTEND_URL=http://159.89.199.144

DEBUG=False
BACKEND_URL=http://backend:8001
```

**Save and exit:** Press `Ctrl+X`, then `Y`, then `Enter`

---

## Step 6: Configure Firewall (1 minute)

```bash
# Allow SSH (IMPORTANT - don't lock yourself out!)
ufw allow 22/tcp

# Allow HTTP
ufw allow 80/tcp

# Allow HTTPS
ufw allow 443/tcp

# Enable firewall
ufw --force enable

# Verify firewall status
ufw status
```

You should see:
```
Status: active

To                         Action      From
--                         ------      ----
22/tcp                     ALLOW       Anywhere
80/tcp                     ALLOW       Anywhere
443/tcp                    ALLOW       Anywhere
```

---

## Step 7: Deploy with Docker (3 minutes)

```bash
# Build and start all services
docker-compose up -d --build

# This will:
# - Build backend (FastAPI) image
# - Build frontend (Django) image
# - Pull MySQL and Nginx images
# - Start all containers
```

**Wait for containers to start** (~2-3 minutes for first build)

```bash
# Check container status
docker-compose ps
```

You should see all containers running:
```
NAME                COMMAND                  STATUS
remhart_backend     "python app/main.py"     Up
remhart_frontend    "python manage.py ru…"   Up
remhart_mysql       "docker-entrypoint.s…"   Up (healthy)
remhart_nginx       "/docker-entrypoint.…"   Up
```

**View logs to verify everything started correctly:**
```bash
docker-compose logs -f
```

Press `Ctrl+C` to stop viewing logs.

---

## Step 8: Train ML Models (3 minutes)

```bash
# Access the backend container
docker-compose exec backend bash

# You're now inside the container
# Run the ML model training script
python setup_ml_models.py
```

This will:
- Generate synthetic training data
- Train all 16 ML models (monitoring, maintenance, energy, decision)
- Save models to `/app/ml_models/trained/`

**Expected output:**
```
======================================================================
 REMHART DIGITAL TWIN - ML MODEL SETUP
======================================================================

[STEP 1/2] Generating training data...
----------------------------------------------------------------------
✓ Generated 10000 training samples

[STEP 2/2] Training all ML models...
----------------------------------------------------------------------
Training Real-time Monitoring Models...
✓ Voltage Anomaly Detection
✓ Harmonic Distortion Detection
✓ Frequency Stability Monitoring
✓ Phase Imbalance Detection

Training Predictive Maintenance Models...
✓ Equipment Failure Prediction
✓ Transformer Overload Prediction
✓ Power Quality Prediction
✓ Voltage Sag Prediction

Training Energy Flow Models...
✓ Load Forecasting
✓ Energy Loss Prediction
✓ Power Flow Optimization
✓ Demand Response Prediction

Training Decision Making Models...
✓ Reactive Power Optimization
✓ Load Balancing
✓ Grid Stability Assessment
✓ Optimal Dispatch

======================================================================
 ✓ ML SETUP COMPLETE!
======================================================================
```

**Keep the container session open - don't exit yet!**

---

## Step 9: Seed Database with Grid Data (2 minutes)

**Still inside the backend container**, run:

```bash
# Seed with 1000 data points (quick start)
python seed_database.py

# OR seed with 5000 data points for better visualization
python seed_database.py --points 5000 --scenario mixed
```

**Expected output:**
```
================================================================================
 REMHART DIGITAL TWIN - DATABASE SEEDING
================================================================================
Initializing database tables...
✓ Database tables created successfully

Generating 5000 data points (mixed scenario)...
  Generated 100/5000 points...
  Generated 200/5000 points...
  ...
  Generated 5000/5000 points...
✓ Generated 5000 data points

Inserting data into database...
  Inserted 100/5000 records...
  Inserted 200/5000 records...
  ...
  Inserted 5000/5000 records...
✓ Successfully inserted 5000 complete data records

Verifying data...
  Timestamps:      5000
  Voltage records: 5000
  Current records: 5000
  Frequency records: 5000
  Active power:    5000
  Reactive power:  5000
✓ Data integrity verified - all counts match!

================================================================================
 ✓ DATABASE SEEDING COMPLETE!
================================================================================

Seeded 5000 data points with 'mixed' scenario.
Your REMHART Digital Twin is ready to use!
```

### Seeding Options:

```bash
# Quick test (1000 points ≈ 50 minutes of data)
python seed_database.py

# Realistic demo (5000 points ≈ 4 hours of data)
python seed_database.py --points 5000

# Comprehensive data (10000 points ≈ 8 hours of data)
python seed_database.py --points 10000

# Specific scenarios:
python seed_database.py --points 3000 --scenario normal          # Stable operation
python seed_database.py --points 3000 --scenario voltage_sag     # Voltage drops
python seed_database.py --points 3000 --scenario overcurrent     # Overload events
python seed_database.py --points 3000 --scenario frequency_drift # Frequency issues
python seed_database.py --points 5000 --scenario mixed          # Mixed anomalies

# View sample data
python seed_database.py --points 100 --sample

# Clear and reseed
python seed_database.py --clear --points 2000
```

**Now exit the container:**
```bash
exit
```

---

## Step 10: Access Your Application ✅

### Open your browser and visit:

```
http://YOUR_DROPLET_IP
```

**Example:**
```
http://159.89.199.144
```

### You should see:

✅ **REMHART Digital Twin Dashboard** with:
- Real-time voltage, current, frequency monitoring
- 3-phase power measurements
- Live charts and graphs
- ML model predictions
- Grid status indicators

---

## Step 11: Verify Everything is Working

### Check API Endpoints:

```
# Health check
http://YOUR_DROPLET_IP/health

# Backend API health
http://YOUR_DROPLET_IP/api/

# ML monitoring predictions
http://YOUR_DROPLET_IP/api/ml/monitoring

# ML maintenance predictions
http://YOUR_DROPLET_IP/api/ml/maintenance

# ML energy flow predictions
http://YOUR_DROPLET_IP/api/ml/energy

# ML decision making
http://YOUR_DROPLET_IP/api/ml/decision
```

### View Real-time Logs:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mysql
docker-compose logs -f nginx
```

Press `Ctrl+C` to stop viewing logs.

---

## Deployment Complete! 🎉

Your REMHART Digital Twin is now:
- ✅ Fully deployed on DigitalOcean
- ✅ Running in Docker containers
- ✅ ML models trained and ready
- ✅ Database seeded with realistic grid data
- ✅ Accessible at `http://YOUR_DROPLET_IP`

---

## Quick Reference Commands

### Restart Services
```bash
cd ~/remhart-digitaltwin
docker-compose restart
```

### Stop Services
```bash
docker-compose down
```

### Start Services
```bash
docker-compose up -d
```

### Update Application
```bash
git pull origin main
docker-compose down
docker-compose up -d --build
```

### View Container Status
```bash
docker-compose ps
```

### Access Backend Container
```bash
docker-compose exec backend bash
```

### Access Frontend Container
```bash
docker-compose exec frontend bash
```

### Access MySQL Database
```bash
docker-compose exec mysql mysql -u root -p
# Enter your MYSQL_ROOT_PASSWORD when prompted
```

### Backup Database
```bash
docker-compose exec mysql mysqldump -u root -p remhart_db > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restore Database
```bash
docker-compose exec -T mysql mysql -u root -p remhart_db < backup_20250202_153045.sql
```

### Monitor Resource Usage
```bash
docker stats
```

### Check Disk Usage
```bash
df -h
docker system df
```

---

## Troubleshooting

### Issue: Containers won't start

```bash
# Check logs for errors
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose up -d --build
```

### Issue: Can't access application

```bash
# Check if containers are running
docker-compose ps

# Check firewall
ufw status

# Check nginx logs
docker-compose logs nginx
```

### Issue: Database connection errors

```bash
# Check if MySQL is healthy
docker-compose exec mysql mysqladmin ping -h localhost

# Check MySQL logs
docker-compose logs mysql

# Verify .env settings
cat .env
```

### Issue: Out of memory

```bash
# Check memory usage
free -h

# Add swap space if needed
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### Issue: Port already in use

```bash
# Check what's using port 80
netstat -tuln | grep :80

# Kill process if needed
fuser -k 80/tcp

# Restart Docker
systemctl restart docker
```

---

## Next Steps

### 1. Set Up Domain Name (Optional)

If you have a domain:

1. Go to DigitalOcean **Networking** → **Domains**
2. Add your domain
3. Create an **A record** pointing to your droplet IP
4. Update `nginx/nginx.conf`:
   - Replace `server_name _;` with `server_name yourdomain.com;`
5. Restart nginx: `docker-compose restart nginx`

### 2. Enable HTTPS/SSL (Recommended)

```bash
# Install Certbot
apt-get install -y certbot python3-certbot-nginx

# Stop nginx container
docker-compose stop nginx

# Obtain SSL certificate
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

### 3. Set Up Automated Backups

```bash
# Create backup script
nano /root/backup_remhart.sh
```

Add this content:
```bash
#!/bin/bash
BACKUP_DIR="/root/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
cd /root/remhart-digitaltwin
docker-compose exec -T mysql mysqldump -u root -p$MYSQL_ROOT_PASSWORD remhart_db > $BACKUP_DIR/remhart_db_$DATE.sql
# Keep only last 7 days of backups
find $BACKUP_DIR -name "remhart_db_*.sql" -mtime +7 -delete
```

Make it executable and schedule:
```bash
chmod +x /root/backup_remhart.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add this line:
0 2 * * * /root/backup_remhart.sh
```

### 4. Monitor Application

Set up DigitalOcean monitoring:
1. Go to your droplet → **Monitoring**
2. Enable alerts for:
   - CPU usage > 80%
   - Memory usage > 90%
   - Disk usage > 85%

### 5. Scale Your Application

For higher traffic:
- **Vertical scaling**: Resize your droplet (more CPU/RAM)
- **Horizontal scaling**: Add load balancer + multiple droplets
- **Database scaling**: Use DigitalOcean Managed MySQL

---

## Cost Estimation

**Monthly DigitalOcean Costs:**

| Resource | Specs | Monthly Cost |
|----------|-------|--------------|
| Droplet (Basic) | 2 vCPU, 4GB RAM | $24 |
| Droplet (Better) | 4 vCPU, 8GB RAM | $48 |
| Backups | 20% of droplet cost | $4.80 - $9.60 |
| Load Balancer (optional) | - | $12 |
| Managed MySQL (optional) | Basic plan | $15 |

**Total:** $24-$100/month depending on configuration

---

## Support & Documentation

- **Full Documentation**: See `DEPLOYMENT.md` in the repository
- **Application Logs**: `docker-compose logs -f`
- **GitHub Issues**: https://github.com/Yasin54386/remhart-digitaltwin/issues

---

## Summary

You've successfully deployed REMHART Digital Twin! 🚀

✅ **4 Containers Running:**
- MySQL database with seeded grid data
- FastAPI backend with 16 trained ML models
- Django frontend with real-time dashboard
- Nginx reverse proxy with security features

✅ **Ready for:**
- Real-time grid monitoring
- ML-powered predictions
- Anomaly detection
- Energy optimization
- Decision support

**Access your dashboard at:** `http://YOUR_DROPLET_IP`

Enjoy monitoring your smart grid! ⚡
