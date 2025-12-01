# Quick Deployment to Digital Ocean

## One-Command Deployment

SSH into your droplet and run:

```bash
ssh root@159.89.199.144
```

Then copy and paste this entire block:

```bash
cd /root && \
git clone https://github.com/Yasin54386/remhart-digitaltwin.git && \
cd remhart-digitaltwin/deploy && \
chmod +x deploy.sh && \
./deploy.sh && \
docker-compose up -d && \
echo "Deployment complete! Access your app at http://159.89.199.144"
```

## View Your Credentials

```bash
cat /root/remhart-digitaltwin/CREDENTIALS.txt
```

## Check Status

```bash
cd /root/remhart-digitaltwin/deploy
docker-compose ps
docker-compose logs -f
```

That's it! Your application will be running at **http://159.89.199.144**

For detailed instructions, see [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
