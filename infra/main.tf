# ─────────────────────────────────────────────────────────────────────────────
# newspaper_ai — Terraform
# Infraestructura completa en Google Cloud con un solo `terraform apply`
#
# Recursos que crea:
#   - Artifact Registry (Docker images)
#   - Secret Manager (GEMINI_API_KEY)
#   - Cloud Run Service (API Gateway)
#   - Cloud Scheduler (generación diaria automática de artículos)
#   - IAM (permisos mínimos)
#
# Uso:
#   cd infra/
#   terraform init
#   terraform plan -var="project_id=TU_PROYECTO" -var="gemini_api_key=AIza..."
#   terraform apply
#
# Variables requeridas: project_id, gemini_api_key
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.7"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Descomenta para guardar el state en GCS (recomendado para trabajo en equipo)
  # backend "gcs" {
  #   bucket = "TU_PROYECTO-terraform-state"
  #   prefix = "newspaper-ai"
  # }
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "GCloud region para Cloud Run"
  type        = string
  default     = "europe-west1"   # Europa — latencia baja desde España
}

variable "gemini_api_key" {
  description = "Gemini API key (AI Studio). Se guarda en Secret Manager."
  type        = string
  sensitive   = true
}

variable "newspaper_name" {
  description = "Nombre del periódico"
  type        = string
  default     = "Nutrición AI"
}

variable "image_tag" {
  description = "Docker image tag a desplegar"
  type        = string
  default     = "latest"
}

# ── Provider ──────────────────────────────────────────────────────────────────

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── APIs habilitadas ──────────────────────────────────────────────────────────
# Habilitar las APIs de GCloud necesarias (sólo la primera vez)

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
# Donde viven los Docker images del proyecto

resource "google_artifact_registry_repository" "newspaper_ai" {
  repository_id = "newspaper-ai"
  format        = "DOCKER"
  location      = var.region
  description   = "Docker images for newspaper_ai"

  depends_on = [google_project_service.apis]
}

locals {
  image_url = "${var.region}-docker.pkg.dev/${var.project_id}/newspaper-ai/newspaper-ai:${var.image_tag}"
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
  account_id   = "newspaper-ai-sa"
  display_name = "newspaper_ai Cloud Run Service Account"
}

# Permiso para leer secrets
resource "google_secret_manager_secret_iam_member" "gemini_key_access" {
  secret_id = google_secret_manager_secret.gemini_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.newspaper_ai.email}"
}

# Permiso para Vertex AI (cuando se migre de GEMINI_API_KEY a Vertex)
resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.newspaper_ai.email}"
}

# Permiso para Cloud Logging
resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.newspaper_ai.email}"
}

# Permiso para Cloud Trace
resource "google_project_iam_member" "trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.newspaper_ai.email}"
}

# ── Cloud Run — API Gateway ───────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "newspaper_ai" {
  name     = "newspaper-ai"
  location = var.region

  template {
    service_account = google_service_account.newspaper_ai.email

    # Recursos: ajustar según presupuesto
    # 1 CPU + 512MB RAM es suficiente para Gemini Flash
    containers {
      image = local.image_url

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        # CPU sólo activa mientras hay requests (máximo ahorro de costes)
        cpu_idle = false
      }

      # Variables de entorno (no sensibles)
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

      # API key desde Secret Manager (nunca hardcodeada)
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

    # Escala a 0 cuando no hay tráfico (máximo ahorro)
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

# Permite acceso público (Lovable llama desde el navegador del usuario)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.newspaper_ai.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Cloud Scheduler — Generación automática diaria ───────────────────────────
# Lanza el pipeline cada mañana a las 7:00 CET (6:00 UTC)
# El periódico tiene artículos frescos cada día sin intervención manual

resource "google_cloud_scheduler_job" "daily_pipeline" {
  name      = "newspaper-ai-daily-pipeline"
  region    = var.region
  schedule  = "0 6 * * *"   # 06:00 UTC = 07:00 CET / 08:00 CEST
  time_zone = "Europe/Madrid"

  http_target {
    uri         = "${google_cloud_run_v2_service.newspaper_ai.uri}/api/pipeline/run"
    http_method = "POST"

    body = base64encode(jsonencode({
      topic_hint   = null
      max_articles = 1
    }))

    headers = {
      "Content-Type" = "application/json"
    }

    # Autenticación: el scheduler se autentica como la SA del proyecto
    oidc_token {
      service_account_email = google_service_account.newspaper_ai.email
      audience              = google_cloud_run_v2_service.newspaper_ai.uri
    }
  }

  retry_config {
    retry_count = 3
  }

  depends_on = [google_cloud_run_v2_service.newspaper_ai]
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "api_url" {
  description = "URL pública de la API — pégala en Lovable como VITE_API_URL"
  value       = google_cloud_run_v2_service.newspaper_ai.uri
}

output "image_url" {
  description = "URL del Docker image en Artifact Registry"
  value       = local.image_url
}

output "docker_push_command" {
  description = "Comando para subir el Docker image"
  value       = "docker push ${local.image_url}"
}

output "docker_build_command" {
  description = "Comando para construir y etiquetar el Docker image"
  value       = "docker build -t ${local.image_url} ."
}
