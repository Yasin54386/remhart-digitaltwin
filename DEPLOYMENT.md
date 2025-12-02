# REMHART Digital Twin - DigitalOcean Deployment Guide

This guide will help you deploy the REMHART Digital Twin application to DigitalOcean using Docker.

## Prerequisites

- DigitalOcean account
- Domain name (optional but recommended)
- SSH access to your server
- Docker and Docker Compose installed on the server

## Deployment Options

### Option 1: DigitalOcean Droplet (Recommended for Full Control)

#### 1. Create a Droplet

1. Log in to DigitalOcean
2. Create a new Droplet:
   - **Distribution**: Ubuntu 22.04 LTS
   - **Plan**: Basic (at minimum 4GB RAM / 2 vCPUs for ML workload)
   - **Datacenter**: Choose closest to your users
   - **Additional Options**: Enable IPv6, Monitoring
3. Add your SSH key
4. Create Droplet

#### 2. Connect to Your Droplet

```bash
ssh root@your-droplet-ip
```

#### 3. Install Docker and Docker Compose

```bash
# Update package list
apt-get update

# Install required packages
apt-get install -y apt-transport-https ca-certificates curl software-properties-common

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

#### 4. Clone Your Repository

```bash
# Install git if not already installed
apt-get install -y git

# Clone your repository
git clone https://github.com/Yasin54386/remhart-digitaltwin.git
cd remhart-digitaltwin
```

#### 5. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your production values
nano .env
```

**Important**: Update these values in `.env`:
- `MYSQL_ROOT_PASSWORD`: Strong password for MySQL root user
- `MYSQL_PASSWORD`: Strong password for application database user
- `SECRET_KEY`: Generate a secure random key (minimum 50 characters)
- `FRONTEND_URL`: Your domain or droplet IP (e.g., http://your-domain.com)
- `DEBUG`: Set to False for production

Generate a secure secret key:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(50))"
```

#### 6. Deploy the Application

```bash
# Build and start all services
docker-compose up -d

# Check if all containers are running
docker-compose ps

# View logs
docker-compose logs -f
```

#### 7. Initialize ML Models

The first time you deploy, you need to train the ML models:

```bash
# Access the backend container
docker-compose exec backend bash

# Run the ML model setup script
python setup_ml_models.py

# Exit the container
exit
```

This will generate synthetic data and train all 16 ML models (takes a few minutes).

#### 8. Configure Firewall

```bash
# Allow SSH
ufw allow 22/tcp

# Allow HTTP and HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Enable firewall
ufw --force enable

# Check status
ufw status
```

#### 9. Access Your Application

Visit `http://your-droplet-ip` in your browser.

### Option 2: DigitalOcean App Platform

App Platform is easier but provides less control. For this application with ML models, a Droplet is recommended.

## Domain Configuration

### 1. Add Domain to DigitalOcean

1. Go to Networking > Domains
2. Add your domain
3. Create an A record pointing to your Droplet IP

### 2. Update Nginx Configuration

Edit `nginx/nginx.conf` and replace `server_name _;` with `server_name your-domain.com;`

### 3. SSL/HTTPS Setup (Let's Encrypt)

```bash
# Install Certbot
apt-get install -y certbot python3-certbot-nginx

# Stop nginx container temporarily
docker-compose stop nginx

# Obtain certificate
certbot certonly --standalone -d your-domain.com -d www.your-domain.com

# Create SSL directory
mkdir -p nginx/ssl

# Copy certificates
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem

# Update nginx.conf to enable HTTPS (uncomment the HTTPS server block)
nano nginx/nginx.conf

# Restart nginx
docker-compose up -d nginx
```

## Maintenance

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mysql
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend
```

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Backup Database

```bash
# Create backup
docker-compose exec mysql mysqldump -u root -p${MYSQL_ROOT_PASSWORD} remhart_db > backup_$(date +%Y%m%d).sql

# Restore from backup
docker-compose exec -T mysql mysql -u root -p${MYSQL_ROOT_PASSWORD} remhart_db < backup_20231201.sql
```

### Monitor Resources

```bash
# Check container resource usage
docker stats

# Check disk usage
df -h

# Check Docker disk usage
docker system df
```

## Scaling

### Vertical Scaling (Increase Droplet Size)

1. Power off the Droplet
2. Resize in DigitalOcean control panel
3. Power on and verify

### Horizontal Scaling (Multiple Droplets)

For high availability:
1. Set up a Load Balancer in DigitalOcean
2. Deploy application on multiple Droplets
3. Configure load balancer to distribute traffic
4. Use managed database (DigitalOcean Managed MySQL)

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs [service-name]

# Rebuild container
docker-compose up -d --build [service-name]
```

### Database Connection Issues

```bash
# Check if MySQL is healthy
docker-compose exec mysql mysqladmin ping -h localhost

# Check backend can connect
docker-compose exec backend python -c "from app.database import engine; print(engine)"
```

### Out of Memory

The ML models require significant RAM. Ensure your Droplet has at least 4GB RAM.

```bash
# Check memory usage
free -h

# If needed, add swap space
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### Port Already in Use

```bash
# Check what's using the port
netstat -tuln | grep :80

# Kill the process if needed
fuser -k 80/tcp
```

## Security Recommendations

1. **Change Default Passwords**: Update all default passwords in `.env`
2. **Enable Firewall**: Use `ufw` to restrict access
3. **SSL/HTTPS**: Use Let's Encrypt for free SSL certificates
4. **Regular Updates**: Keep Docker images and system packages updated
5. **Backup Strategy**: Implement automated database backups
6. **Monitoring**: Set up DigitalOcean monitoring and alerts
7. **Rate Limiting**: Nginx configuration includes rate limiting
8. **Secrets Management**: Never commit `.env` to version control

## Performance Optimization

1. **Use managed database**: Consider DigitalOcean Managed MySQL for better performance
2. **Enable caching**: Add Redis for API response caching
3. **CDN**: Use DigitalOcean Spaces + CDN for static assets
4. **Database indexing**: Optimize queries with proper indexes
5. **ML model caching**: Models are loaded once and cached in memory

## Cost Estimation

Approximate monthly costs on DigitalOcean:

- **Droplet (4GB/2vCPU)**: $24/month
- **Managed Database (optional)**: $15/month (basic)
- **Load Balancer (optional)**: $12/month
- **Backups**: $4.80/month (20% of Droplet cost)
- **Domain**: $12-15/year

**Total**: ~$24-60/month depending on options

## Support

For issues related to:
- **Application bugs**: Open an issue on GitHub
- **DigitalOcean infrastructure**: Contact DigitalOcean support
- **Docker/deployment**: Check Docker documentation

## Next Steps

After successful deployment:
1. Test all API endpoints
2. Verify ML model predictions are working
3. Set up monitoring and alerts
4. Configure automated backups
5. Plan for scaling based on traffic
