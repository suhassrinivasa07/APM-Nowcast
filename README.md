# APM-Nowcast

> **An Uncertainty-Aware Physics-Guided Adaptive Multimodal Deep Learning Framework for Thunderstorm Initiation and Lightning Nowcasting over India**

APM-Nowcast is an end-to-end meteorological deep learning platform engineered to forecast Thunderstorm Initiation (TI) and nowcast lightning strikes with high spatiotemporal fidelity across the Indian subcontinent.

The framework integrates multimodal observations (INSAT-3D/3DR geostationary satellite imagery, IMD Doppler Weather Radars, and IITM Lightning Location Networks) paired with thermodynamic convective proxies and uncertainty quantification.

---

## Hardware Target & Specifications

- **OS:** Windows 11
- **Compute:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Frameworks:** Python 3.11+, PyTorch, FastAPI, React 18, TypeScript, Vite
- **VRAM Optimizations:** FP16 mixed precision, gradient checkpointing, and spatial tiling.

---

## Repository Structure

```text
APM-Nowcast/
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions (data, weights, environments)
├── README.md                 # Project documentation and quickstart
├── backend/                  # FastAPI Application & Scientific Computing
│   ├── app/
│   │   ├── api/              # API router and versioned endpoints (v1)
│   │   │   └── v1/
│   │   │       ├── endpoints/
│   │   │       │   └── health.py  # GET /health & /api/v1/health
│   │   │       └── router.py
│   │   ├── core/             # Configuration & environment settings
│   │   │   └── config.py
│   │   ├── db/               # PostgreSQL persistence (future phase)
│   │   ├── schemas/          # Pydantic data validation schemas
│   │   │   └── health.py
│   │   ├── services/         # Physics engines & ML inference pipelines
│   │   └── main.py           # FastAPI entrypoint & CORS configuration
│   ├── tests/                # Automated pytest suite
│   │   ├── conftest.py       # Test fixtures and client initialization
│   │   └── test_health.py    # Health check and endpoint assertions
│   ├── pyproject.toml        # Modern Python package configuration
│   └── requirements.txt      # Backend Python dependencies
├── frontend/                 # React + TypeScript + Vite Dashboard
│   ├── src/
│   │   ├── api/              # Backend API client
│   │   │   └── client.ts
│   │   ├── components/       # UI Components
│   │   │   └── HealthStatus.tsx  # Live backend status indicator
│   │   ├── App.tsx           # Dashboard layout
│   │   ├── main.tsx          # React application root
│   │   └── index.css         # Styling
│   ├── index.html            # HTML entrypoint
│   ├── package.json          # Node dependencies & scripts
│   ├── tsconfig.json         # TypeScript configuration
│   └── vite.config.ts        # Vite build tool configuration
├── data/                     # Data stores (excluded from git)
│   ├── raw/                  # Raw INSAT, DWR, and Lightning feeds
│   └── processed/            # Collocated grids and NetCDF/Zarr tensors
├── deployment/               # Containerization & Deployment
│   ├── Dockerfile.backend    # Backend container definition
│   ├── Dockerfile.frontend   # Frontend container definition
│   └── docker-compose.yml    # Multi-container orchestration
└── research/                 # Scientific Notebooks & EDA
    └── notebooks/            # Exploratory research notebooks
```

---

## Getting Started

### 1. Environment Configuration

Copy the example configuration to `.env`:

```powershell
Copy-Item .env.example .env
```

### 2. Backend Setup & Execution

#### Activate the Virtual Environment:
```powershell
backend\.venv\Scripts\Activate.ps1
```

*(Or on Bash: `source backend/.venv/bin/activate`)*

#### Install Dependencies:
```powershell
pip install -r backend/requirements.txt
```

#### Run the Backend Development Server:
```powershell
uvicorn app.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000
```

- **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

#### Run Backend Automated Tests:
```powershell
backend\.venv\Scripts\pytest.exe backend/tests -v
```

---

### 3. Frontend Setup & Execution

#### Install Node Dependencies:
```powershell
cd frontend
npm install
```

#### Run the Frontend Development Server:
```powershell
npm run dev
```

The frontend dashboard will be available at [http://localhost:5173](http://localhost:5173).

#### Build for Production:
```powershell
npm run build
```

---

### 4. Docker Deployment (Optional)

To spin up the containerized stack:

```powershell
docker-compose -f deployment/docker-compose.yml up --build
```

---

## 7-Day Research Prototype Roadmap

| Day | Focus | Milestone |
|---|---|---|
| **Day 1** | **Project Foundation** | Modular architecture, FastAPI /health, React+TS frontend, test suite. |
| **Day 2** | **Data Ingestion & Preprocessing** | INSAT-3D/3DR TIR & DWR grid collocation into normalized Xarray datasets. |
| **Day 3** | **Physics-Guided Feature Engineering** | Instability indices (CAPE/CIN proxy), cloud-top cooling rate, echo-top height. |
| **Day 4** | **Lightweight Multimodal Architecture** | Dual-branch ConvNeXt/U-Net backbone optimized for 4GB VRAM. |
| **Day 5** | **Uncertainty Quantification & Training** | Evidential Deep Learning / MC Dropout for calibrated lightning risk. |
| **Day 6** | **Inference Pipeline & Backend Integration** | Streaming nowcast generator (0-2 hour lead time) with GeoJSON/COG output. |
| **Day 7** | **Interactive Visualization & Evaluation** | Leaflet radar overlay, CSI/HSS verification metrics, end-to-end demo. |
