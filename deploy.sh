#!/bin/bash

set -e

# Configuration
BACKUP_DIR="/var/backups/betabank"
LOG_FILE="/var/log/betabank/deploy.log"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Create necessary directories
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if required files exist
if [ ! -f "$COMPOSE_FILE" ]; then
    log "Error: $COMPOSE_FILE not found"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    log "Error: $ENV_FILE not found"
    exit 1
fi

# Create database backup
log "Creating database backup..."
if docker-compose ps | grep -q db; then
    BACKUP_FILE="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"
    docker-compose exec -T db pg_dump -U betabank betabank > "$BACKUP_FILE"
    log "Database backup created at $BACKUP_FILE"
else
    log "Database container not running, skipping backup"
fi

# Pull latest changes
log "Pulling latest changes..."
git pull

# Build and deploy
log "Building and deploying services..."
docker-compose pull
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be healthy
log "Waiting for services to be healthy..."
timeout 300 bash -c '
until docker-compose ps | grep "healthy" | wc -l | grep -q "3"; do
    echo "Waiting for all services to be healthy..."
    sleep 5
done
'

# Run database migrations
log "Running database migrations..."
docker-compose exec -T backend alembic upgrade head

log "Deployment completed successfully!"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "backup_*.sql" -type f -mtime +7 -delete

# Print service status
log "Current service status:"
docker-compose ps 