# Deployment Configurations

> [!NOTE]
> These deployment configurations (Render Blueprint, Netlify configuration, Vercel routes, Dockerfile, and Docker Compose) are preserved from the original base project for future reference. They are **not in use yet** for local development.

### Preserved Files
- `render.yaml`: Render Blueprint specification for deploying the FastAPI backend.
- `netlify.toml`: Netlify configuration for serving the static frontend with security headers.
- `vercel.json`: Vercel static build routing and CORS configuration.
- `Dockerfile`: Multi-stage Docker container build for the FastAPI backend.
- `docker-compose.yml`: Local container composition.
- `docker-entrypoint.sh`: Container entrypoint script that handles database initialization before running uvicorn.
- `.dockerignore`: Files excluded from container build context.
