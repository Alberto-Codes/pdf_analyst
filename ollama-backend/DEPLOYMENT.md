# Cloud Run Deployment Guide for PDF Analyst - Ollama LLM Backend

This guide provides instructions for deploying the Ollama LLM backend component of the PDF Analyst application as a Cloud Run service on Google Cloud Platform (GCP). This backend serves Llama 3.2 to answer questions about PDFs using PHI data agents.

## Scope

This deployment guide covers:
- Only the Ollama LLM backend component
- Deployment of the Cloud Run service that serves Llama 3.2
- Authentication setup for the main PDF Analyst service to call this backend

This guide does not cover:
- Deployment of the main PDF Analyst application
- PHI data workflow components and their deployment
- Integration between components

All files referenced in this guide should be located in the `/ollama-backend` directory at the project root, separate from the PHI data workflow code.

## Project Structure
```
project-root/
├── ollama-backend/      # Location of this component
│   ├── Dockerfile
│   └── DEPLOYMENT.md
├── [other-directories]  # Other PDF Analyst components
```

[Rest of content remains the same from Prerequisites onwards...]

## Prerequisites

Before beginning the deployment process, ensure you have:

- Google Cloud SDK installed and properly configured
- An active Google Cloud project with billing enabled
- Appropriate permissions to create and manage GCP resources
- Docker installed locally (if testing locally before deployment)

## Environment Setup

First, set up your environment variables. Create a `.env` file or export these variables in your terminal:

```sh
export PROJECT_ID=your-project-id
export REPOSITORY=your-repository
export SERVICE_ACCOUNT_NAME=your-service-account-name
export REGION=us-central1
```

## Deployment Steps

### 1. Create an Artifact Registry Repository

Create a Docker repository in Artifact Registry to store your container images:

```sh
gcloud artifacts repositories create $REPOSITORY \
    --repository-format=docker \
    --location=$REGION \
    --description="Repository for PDF Analyst container images"
```

### 2. Service Account Configuration

Create and configure a service account for the Cloud Run service:

```sh
# Create the service account
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="PDF Analyst Cloud Run Service Account"
```

### 3. Build and Push the Container Image

Navigate to your Dockerfile directory and build the image:

```sh
# Navigate to Dockerfile location
cd docker-file

# Build and push using Cloud Build
gcloud builds submit \
    --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/ollama-llama3 \
    --machine-type e2-highcpu-32 \
    --timeout=2h
```

### 4. Deploy to Cloud Run

Deploy your service with optimized configurations:

```sh
gcloud beta run deploy ollama-llama3 \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/ollama-llama3 \
    --region $REGION \
    --concurrency 4 \
    --cpu 8 \
    --set-env-vars OLLAMA_NUM_PARALLEL=4 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --max-instances 7 \
    --memory 32Gi \
    --no-allow-unauthenticated \
    --no-cpu-throttling \
    --service-account $SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com \
    --timeout=600
```

## Security Configuration

### Authentication Setup

To allow your application to call this Cloud Run endpoint:

1. Navigate to the Cloud Run service in the Google Cloud Console
2. Go to "Permissions" tab
3. Click "Add Principal" 
4. Add your calling application's service account (this is different from the `$SERVICE_ACCOUNT_NAME` used to run the Cloud Run service)
5. Grant it the "Cloud Run Invoker" role (roles/run.invoker)

Alternatively, you can use this command to grant access:

```sh
# Grant Cloud Run Invoker role to your calling application's service account
gcloud run services add-iam-policy-binding ollama-llama3 \
    --member="serviceAccount:your-calling-application-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --region=$REGION
```

Note: Replace `your-calling-application-sa` with the service account name of the application that will be calling this Cloud Run endpoint. This is separate from the `$SERVICE_ACCOUNT_NAME` that the Cloud Run service itself runs as.

## Resource Configuration Details

The deployment uses the following resource allocations:

- CPU: 8 cores (no throttling)
- Memory: 32GB
- GPU: 1x NVIDIA L4
- Concurrency: 4 requests per instance
- Maximum instances: 7
- Request timeout: 600 seconds

## Important Notes

- Monitor your resource usage and adjust configurations based on workload demands
- Regular backups of configuration and environment variables are recommended
- Keep your service account credentials secure and rotate them periodically
- Consider implementing monitoring and logging solutions
- Test your deployment in a staging environment before production

## Troubleshooting

If you encounter deployment issues:

1. Check Cloud Build logs for build failures
2. Verify service account permissions
3. Ensure GPU quotas are sufficient in your region
4. Review Cloud Run logs for runtime errors

## Cost Management

To optimize costs:

- Monitor instance scaling patterns
- Adjust max instances based on usage patterns
- Consider using spot instances for non-critical workloads
- Set up budget alerts in Google Cloud Console