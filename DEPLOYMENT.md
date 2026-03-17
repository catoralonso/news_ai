# newspaper_ai — Guía de Despliegue y Conexión con Lovable

## Qué hay en esta guía

1. Estructura de archivos nuevos
2. Cómo añadir los archivos al repo
3. Despliegue en Cloud Run (manual, sin Terraform)
4. Despliegue con Terraform
5. Conectar la API con Lovable
6. Cloud Logging y Cloud Trace — cómo leerlos
7. Flujo completo en producción

---

## 1. Archivos nuevos a añadir al repo

```
newspaper_ai/
├── Dockerfile             ← build del contenedor
├── api/
│   ├── __init__.py        ← vacío
│   └── main.py            ← FastAPI gateway (reemplaza app.py)
├── infra/
│   └── main.tf            ← Terraform (infraestructura completa)
└── config.py              ← actualizado con setup_observability()
```

`app.py` (Streamlit) queda en el repo pero ya no se usa en producción.

---

## 2. Añadir al repo

```bash
mkdir -p api infra
touch api/__init__.py

# Copiar los archivos generados
cp Dockerfile .
cp api_main.py api/main.py
cp config_new.py config.py
cp main.tf infra/main.tf

git add Dockerfile api/ infra/ config.py
git commit -m "feat: FastAPI gateway + Dockerfile + Terraform + Cloud Observability"
git push
```

---

## 3. Despliegue en Cloud Run — Manual (recomendado para empezar)

### Paso 1 — Configurar proyecto GCloud

```bash
gcloud config set project TU_PROYECTO_ID
gcloud config set run/region europe-west1
```

### Paso 2 — Habilitar APIs necesarias

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudtrace.googleapis.com \
  logging.googleapis.com
```

### Paso 3 — Guardar la API key en Secret Manager

```bash
# Nunca en variables de entorno directas en producción
echo -n "TU_GEMINI_API_KEY" | \
  gcloud secrets create gemini-api-key \
    --data-file=- \
    --replication-policy=automatic
```

### Paso 4 — Crear Artifact Registry

```bash
gcloud artifacts repositories create newspaper-ai \
  --repository-format=docker \
  --location=europe-west1 \
  --description="Docker images for newspaper_ai"
```

### Paso 5 — Build y push del Docker image

```bash
# Autenticar Docker con GCloud
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Build
docker build -t europe-west1-docker.pkg.dev/TU_PROYECTO/newspaper-ai/newspaper-ai:latest .

# Push
docker push europe-west1-docker.pkg.dev/TU_PROYECTO/newspaper-ai/newspaper-ai:latest
```

### Paso 6 — Deploy en Cloud Run

```bash
gcloud run deploy newspaper-ai \
  --image europe-west1-docker.pkg.dev/TU_PROYECTO/newspaper-ai/newspaper-ai:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars NEWSPAPER_NAME="Nutrición AI",REGION_NEWS=ES,CHAT_MODEL=gemini-2.0-flash \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

Al terminar, GCloud imprime la URL del servicio:
```
Service URL: https://newspaper-ai-XXXXX-ew.a.run.app
```

**Esa URL es la que pegas en Lovable.**

### Paso 7 — Verificar que funciona

```bash
curl https://newspaper-ai-XXXXX-ew.a.run.app/health
# → {"status":"ok","newspaper":"Nutrición AI",...}

curl https://newspaper-ai-XXXXX-ew.a.run.app/docs
# → Swagger UI con todos los endpoints
```

---

## 4. Despliegue con Terraform (opcional, para la presentación)

```bash
cd infra/

# Inicializar (descarga el provider de Google)
terraform init

# Ver qué va a crear sin aplicar
terraform plan \
  -var="project_id=TU_PROYECTO_ID" \
  -var="gemini_api_key=TU_API_KEY"

# Crear toda la infraestructura (Cloud Run + Scheduler + Secrets + IAM)
terraform apply \
  -var="project_id=TU_PROYECTO_ID" \
  -var="gemini_api_key=TU_API_KEY"
```

Terraform imprime al final:
```
api_url = "https://newspaper-ai-XXXXX-ew.a.run.app"
docker_build_command = "docker build -t ..."
docker_push_command  = "docker push ..."
```

Para destruir toda la infraestructura:
```bash
terraform destroy -var="project_id=..." -var="gemini_api_key=..."
```

---

## 5. Conectar la API con Lovable

En el proyecto de Lovable, ve a **Settings → Environment Variables** y añade:

```
VITE_API_URL=https://newspaper-ai-XXXXX-ew.a.run.app
```

En el código React de Lovable, todas las llamadas a la API deben usar esta variable:

```javascript
const API = import.meta.env.VITE_API_URL;

// Ejemplo: obtener artículos
const res = await fetch(`${API}/api/articles`);
const { articles } = await res.json();

// Ejemplo: lanzar pipeline
const res = await fetch(`${API}/api/pipeline/run`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ topic_hint: null, max_articles: 1 })
});
const { job_id } = await res.json();

// Ejemplo: polling de estado del pipeline
const check = await fetch(`${API}/api/pipeline/status/${job_id}`);
const { status, result } = await check.json();

// Ejemplo: chatbot Mauro con streaming
const res = await fetch(`${API}/api/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Hola Mauro', session_id: 'user_123' })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value);
  const lines = buffer.split('\n\n');
  buffer = lines.pop();
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const chunk = JSON.parse(line.slice(6));
      if (!chunk.done) {
        // añadir chunk.text al estado del mensaje
        setMessage(prev => prev + chunk.text);
      }
    }
  }
}
```

### Endpoints disponibles para Lovable

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Estado del servidor |
| GET | `/api/trends` | Tendencias del día (José) |
| POST | `/api/pipeline/run` | Lanzar generación de artículo |
| GET | `/api/pipeline/status/{job_id}` | Estado del pipeline |
| GET | `/api/articles` | Lista artículos generados |
| GET | `/api/articles/{id}` | Artículo completo |
| GET | `/api/social/{article_id}` | Posts de redes sociales |
| POST | `/api/chat` | Chatbot Mauro (SSE streaming) |
| GET | `/docs` | Swagger UI interactivo |

---

## 6. Cloud Logging y Cloud Trace

### Ver logs en tiempo real

```bash
# Terminal
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=newspaper-ai"

# O en la consola web:
# Google Cloud Console → Logging → Logs Explorer
# Filtro: resource.type="cloud_run_revision"
```

Cualquier `logging.info(...)` en cualquier agente aparece aquí automáticamente.

### Ver traces

```
Google Cloud Console → Trace → Trace List
```

Cada request a `/api/pipeline/run` genera un trace completo mostrando cuánto tarda cada paso.

### Ver métricas de Cloud Run

```
Google Cloud Console → Cloud Run → newspaper-ai → Metrics
```

Muestra: request count, latencia, errores, instancias activas.

---

## 7. Flujo completo en producción

```
07:00 CET (cada día)
  Cloud Scheduler dispara POST /api/pipeline/run
       ↓
  Cloud Run despierta (cold start ~3s si estaba a 0)
       ↓
  José + Camila (paralelo, asyncio.gather)
       ↓
  Manuel genera el artículo
       ↓
  Asti genera social media pack
       ↓
  Artículo guardado en data/articles/
  Social pack guardado en data/social_media_output/
       ↓
  Cloud Logging registra todo el proceso
  Cloud Trace muestra el timeline completo
       ↓
  Lovable lee GET /api/articles y muestra el artículo nuevo
```

El equipo puede ver los logs desde la consola de GCloud sin necesidad de abrir ningún terminal.

---

## Notas importantes

- **Cold start**: Con `min-instances=0` la primera request del día tarda ~5-10s en arrancar. 
  Para la exposición, usa `min-instances=1` para que siempre esté caliente.
  
- **Datos efímeros**: Cloud Run no tiene disco persistente entre reinicios.
  Para producción real, conectar a Cloud Storage o Firestore para guardar artículos.
  
- **ChromaDB en Cloud Run**: ChromaDB escribe en disco. Funciona para la demo,
  pero en producción migrar a Vertex AI Vector Search o usar ChromaDB con GCS backend.

- **`.env` nunca en el repo**: Los secrets van siempre en Secret Manager.
  Verificar que `.gitignore` incluye `.env`.
