# Deployment Guide for BetaBank

This guide explains how to deploy the BetaBank application using Docker and Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- Git
- A server with at least 2GB RAM
- Domain name (optional)

## Local Deployment

1. Clone the repository:
```bash
git clone <your-repo-url>
cd betabank
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Edit the `.env` file with your configuration:
- Generate a secure SECRET_KEY
- Set NEXT_PUBLIC_API_URL to your backend URL
- Configure AWS credentials if using S3 for datasets

4. Build and start the containers:
```bash
docker-compose up -d --build
```

5. Run database migrations:
```bash
docker-compose exec backend alembic upgrade head
```

The application should now be running at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Database: localhost:5432

## Production Deployment

For production deployment, additional steps are recommended:

1. Use a reverse proxy (like Nginx) for SSL termination and routing
2. Set up proper database backups
3. Use production-grade PostgreSQL settings
4. Configure proper logging
5. Set up monitoring

### Example Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Security Considerations

1. Update the SECRET_KEY in production
2. Use strong database passwords
3. Configure proper firewall rules
4. Enable rate limiting
5. Set up proper CORS configuration
6. Use SSL/TLS certificates
7. Implement proper backup strategy

### Database Backups

Set up automated backups:

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/path/to/backups"
docker-compose exec -T db pg_dump -U betabank betabank > "$BACKUP_DIR/backup_$TIMESTAMP.sql"
EOF

# Make it executable
chmod +x backup.sh

# Add to crontab (daily at 2 AM)
0 2 * * * /path/to/backup.sh
```

### Monitoring

Consider setting up:
1. Prometheus for metrics
2. Grafana for visualization
3. Alert manager for notifications
4. Log aggregation (e.g., ELK stack)

## Scaling

For scaling the application:

1. Use a managed database service
2. Set up load balancing
3. Use container orchestration (e.g., Kubernetes)
4. Implement caching (Redis)
5. Use CDN for static assets

## Troubleshooting

Common issues and solutions:

1. Database connection issues:
   - Check if PostgreSQL is running
   - Verify connection string
   - Check network connectivity

2. Frontend can't connect to backend:
   - Verify API URL configuration
   - Check CORS settings
   - Verify network connectivity

3. Container issues:
   - Check logs: `docker-compose logs`
   - Verify resource availability
   - Check container health

## Maintenance

Regular maintenance tasks:

1. Update dependencies regularly
2. Monitor disk space
3. Review and rotate logs
4. Check and update SSL certificates
5. Review security updates
6. Monitor database performance

## Rollback Procedure

In case of deployment issues:

1. Keep the previous version tagged in Docker
2. Maintain database migration history
3. Document application state dependencies
4. Have a tested rollback plan 