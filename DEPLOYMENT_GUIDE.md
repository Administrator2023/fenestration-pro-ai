# 🚀 Deployment Guide - Fenestration Pro AI Complete SOTA Edition

## 📋 Overview

This guide covers deployment options for the complete state-of-the-art Fenestration Pro AI system with all advanced features.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                     │
├─────────────────────────────────────────────────────────────┤
│  Advanced RAG  │  Multimodal   │  Analytics  │  Enterprise  │
│   Pipeline     │  Processing   │  Dashboard  │   Features   │
├─────────────────────────────────────────────────────────────┤
│              Performance Optimization Layer                  │
├─────────────────────────────────────────────────────────────┤
│  Vector Stores │  Databases   │   Caching   │   Monitoring  │
│  (Chroma/FAISS)│ (SQLite/PG)  │  (Redis)    │  (Analytics)  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Deployment

### Option 1: Local Development

```bash
# 1. Clone and setup
git clone https://github.com/administrator2023/fenestration-pro-ai.git
cd fenestration-pro-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
echo 'OPENAI_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml

# 4. Launch application
streamlit run fenestration_pro_ai_complete.py
```

### Option 2: Docker Deployment

```dockerfile
# Dockerfile (create this file)
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "fenestration_pro_ai_complete.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run Docker container
docker build -t fenestration-pro-ai .
docker run -p 8501:8501 -e OPENAI_API_KEY="your-api-key" fenestration-pro-ai
```

### Option 3: Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:password@postgres:5432/fenestration_ai
    volumes:
      - ./vector_stores:/app/vector_stores
      - ./analytics.db:/app/analytics.db
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:password@postgres:5432/fenestration_ai
    command: python -c "from enterprise_features import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=fenestration_ai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

```bash
# Deploy with Docker Compose
docker-compose up -d
```

## ☁️ Cloud Deployment

### Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy complete SOTA application"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file: `fenestration_pro_ai_complete.py`
   - Add secrets in dashboard:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```

### AWS Deployment

#### ECS (Elastic Container Service)

```bash
# 1. Build and push to ECR
aws ecr create-repository --repository-name fenestration-pro-ai
docker tag fenestration-pro-ai:latest <account-id>.dkr.ecr.<region>.amazonaws.com/fenestration-pro-ai:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/fenestration-pro-ai:latest

# 2. Create ECS task definition
# 3. Create ECS service
# 4. Configure load balancer
```

#### EC2 Deployment

```bash
# 1. Launch EC2 instance (t3.large recommended)
# 2. Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# 3. Deploy application
docker run -d -p 80:8501 \
  -e OPENAI_API_KEY="your-api-key" \
  --restart unless-stopped \
  fenestration-pro-ai:latest
```

### Google Cloud Platform

#### Cloud Run

```bash
# 1. Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/fenestration-pro-ai

# 2. Deploy to Cloud Run
gcloud run deploy fenestration-pro-ai \
  --image gcr.io/PROJECT_ID/fenestration-pro-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY="your-api-key"
```

#### GKE (Google Kubernetes Engine)

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fenestration-pro-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fenestration-pro-ai
  template:
    metadata:
      labels:
        app: fenestration-pro-ai
    spec:
      containers:
      - name: app
        image: gcr.io/PROJECT_ID/fenestration-pro-ai:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: fenestration-pro-ai-service
spec:
  selector:
    app: fenestration-pro-ai
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

### Azure Deployment

#### Container Instances

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name fenestration-pro-ai \
  --image fenestration-pro-ai:latest \
  --dns-name-label fenestration-pro-ai \
  --ports 8501 \
  --environment-variables OPENAI_API_KEY="your-api-key"
```

## 🔧 Production Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/db
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=200
ENABLE_ANALYTICS=true
ENABLE_MULTIMODAL=true
```

### Performance Tuning

```python
# config/production.py
STREAMLIT_CONFIG = {
    "server.maxUploadSize": 200,
    "server.maxMessageSize": 200,
    "server.enableCORS": False,
    "server.enableXsrfProtection": True,
    "browser.gatherUsageStats": False,
    "global.developmentMode": False
}

RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_retrieval_docs": 5,
    "cache_ttl": 3600
}

PERFORMANCE_CONFIG = {
    "max_workers": 16,
    "enable_caching": True,
    "cache_size": 1000,
    "async_processing": True
}
```

### Security Configuration

```python
# security/config.py
SECURITY_CONFIG = {
    "enable_authentication": True,
    "jwt_secret_key": "your-secret-key",
    "password_min_length": 8,
    "rate_limit_per_hour": 1000,
    "max_file_size_mb": 200,
    "allowed_file_types": ["pdf", "docx", "txt"]
}
```

## 📊 Monitoring & Logging

### Application Monitoring

```python
# monitoring/setup.py
import logging
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
ACTIVE_USERS = Gauge('active_users', 'Active users')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks

```python
# health/checks.py
async def health_check():
    """Comprehensive health check"""
    checks = {
        "database": check_database_connection(),
        "redis": check_redis_connection(),
        "openai_api": check_openai_api(),
        "vector_store": check_vector_store(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage()
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy Fenestration Pro AI

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/ -v --cov=./ --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to Streamlit Cloud
      run: |
        # Deployment script here
        echo "Deploying to production..."
```

## 📈 Scaling Considerations

### Horizontal Scaling

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fenestration-pro-ai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fenestration-pro-ai
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancing

```nginx
# nginx.conf
upstream fenestration_app {
    server app1:8501;
    server app2:8501;
    server app3:8501;
}

server {
    listen 80;
    server_name fenestration-pro-ai.com;

    location / {
        proxy_pass http://fenestration_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🔒 Security Best Practices

### SSL/TLS Configuration

```bash
# Let's Encrypt SSL
certbot --nginx -d fenestration-pro-ai.com
```

### API Security

```python
# security/middleware.py
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Security(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

## 📋 Maintenance

### Backup Strategy

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)

# Backup vector stores
tar -czf "backup_vectors_$DATE.tar.gz" vector_stores/

# Backup database
pg_dump fenestration_ai > "backup_db_$DATE.sql"

# Backup to S3
aws s3 cp "backup_vectors_$DATE.tar.gz" s3://fenestration-backups/
aws s3 cp "backup_db_$DATE.sql" s3://fenestration-backups/
```

### Update Procedure

```bash
#!/bin/bash
# update.sh
echo "Starting update procedure..."

# 1. Backup current state
./backup.sh

# 2. Pull latest code
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt

# 4. Run database migrations
python migrate.py

# 5. Restart services
docker-compose restart

echo "Update completed successfully!"
```

## 🎯 Performance Benchmarks

### Expected Performance

- **Response Time**: < 3 seconds for RAG queries
- **Document Processing**: < 30 seconds per PDF
- **Concurrent Users**: 100+ with proper scaling
- **Memory Usage**: ~2GB per instance
- **CPU Usage**: ~50% under normal load

### Optimization Tips

1. **Enable Redis caching** for better performance
2. **Use SSD storage** for vector databases
3. **Configure proper resource limits** in containers
4. **Monitor and tune** chunk sizes based on your documents
5. **Use CDN** for static assets

## 🆘 Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```bash
   # Increase container memory
   docker run -m 4g fenestration-pro-ai
   ```

2. **Slow Performance**:
   ```bash
   # Enable caching
   export REDIS_URL=redis://localhost:6379
   ```

3. **API Rate Limits**:
   ```python
   # Implement exponential backoff
   @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
   def call_openai_api():
       # API call here
   ```

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/administrator2023/fenestration-pro-ai/wiki)
- **Issues**: [GitHub Issues](https://github.com/administrator2023/fenestration-pro-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/administrator2023/fenestration-pro-ai/discussions)
- **Email**: support@fenestrationpro.ai

---

**🚀 Ready to deploy? Choose your preferred method above and launch your state-of-the-art AI system!**