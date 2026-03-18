# ─────────────────────────────────────────────────────────────────────────────
# newspaper_ai — Terraform
# Complete infraestructure for Google Cloud using `terraform apply`
#
# Resources created:
#   - Artifact Registry (Docker images)
#   - Secret Manager (GEMINI_API_KEY)
#   - Cloud Run Service (API Gateway)
#   - IAM (permisos mínimos)
#
# Use:
#   cd infra/
#   terraform init
#   terraform plan -var="project_id=TU_PROYECTO" -var="gemini_api_key=AIza..."
#   terraform apply
#
# Variables needed: project_id, gemini_api_key
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Uncoment to save state on GCS
  # backend "gcs" {
  #   bucket = "YOUR-PROJECT-terraform-state"
  #   prefix = "newspaper-ai"
  # }
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "GCloud region for Cloud Run"
  type        = string
  default     = "us-central1"
}

variable "gemini_api_key" {
  description = "Gemini API key (AI Studio). Saved in Secret Manager."
  type        = string
  sensitive   = true
}

variable "newspaper_name" {
  description = "Newspaper name"
  type        = string
  default     = "Savia"
}

variable "image_tag" {
  description = "Docker image tag for deploy"
  type        = string
  default     = "latest"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "savia"
}

# ── Provider ──────────────────────────────────────────────────────────────────

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── APIs enabeling ──────────────────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "aiplatform.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# ── Artifact Registry ─────────────────────────────────────────────────────────

resource "google_artifact_registry_repository" "newspaper_ai" {
  repository_id = var.service_name
  format        = "DOCKER"
  location      = var.region
  description   = "Docker images for newspaper_ai"

  depends_on = [google_project_service.apis]
}

locals {
  image_url = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.service_name}:${var.image_tag}"
}

# ── Secret Manager — GEMINI_API_KEY ───────────────────────────────────────────

resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "gemini_api_key" {
  secret      = google_secret_manager_secret.gemini_api_key.id
  secret_data = var.gemini_api_key
}

# ── Service Account para Cloud Run ────────────────────────────────────────────

resource "google_service_account" "newspaper_ai" {
  account_id   = "${var.service_name}-sa"
  display_name = "${var.service_name} Cloud Run Service Account"
}

# Permissions to read secrets
resource "google_secret_manager_secret_iam_member" "gemini_key_access" {
  secret_id = google_secret_manager_secret.gemini_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.newspaper_ai.email}"
}

# Permissions for Vertex AI
resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.newspaper_ai.email}"
}

# Permissions for Cloud Logging
resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.newspaper_ai.email}"
}

# Permissions for Cloud Trace
resource "google_project_iam_member" "trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.newspaper_ai.email}"
}

# ── Cloud Run — API Gateway ───────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "newspaper_ai" {
  name     = var.service_name
  location = var.region

  template {
    service_account = google_service_account.newspaper_ai.email

    # Resources: adjust according to budget
    containers {
      image = local.image_url

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        # CPU ative only on request
        cpu_idle = false
      }

      # Environment Variables
      env {
        name  = "NEWSPAPER_NAME"
        value = var.newspaper_name
      }
      env {
        name  = "REGION_NEWS"
        value = "ES"
      }
      env {
        name  = "CHAT_MODEL"
        value = "gemini-2.0-flash"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_REGION"
        value = var.region
      }

      # API key from Secret Manager
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      # Health check endpoint
      liveness_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 15
        period_seconds        = 30
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        failure_threshold     = 10
        period_seconds        = 5
      }
    }

    # Scaling to 0 when there is no traffic
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  depends_on = [
    google_project_service.apis,
    google_artifact_registry_repository.newspaper_ai,
    google_secret_manager_secret_version.gemini_api_key,
  ]
}

# Allows public access
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.newspaper_ai.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "api_url" {
  description = "Public URL from API - for Lovable (VITE_API_URL)"
  value       = google_cloud_run_v2_service.newspaper_ai.uri
}

output "image_url" {
  description = "URL of the Docker image on the Artifact Registry"
  value       = local.image_url
}

output "docker_push_command" {
  description = "To upload the Docker image"
  value       = "docker push ${local.image_url}"
}

output "docker_build_command" {
  description = "To build and tag Docker image"
  value       = "docker build -t ${local.image_url} ."
}
