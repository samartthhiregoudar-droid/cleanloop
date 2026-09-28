"""CleanLoop Backend Server (FastAPI + WebSockets + SQLite)."""
import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.database import get_analytics, get_incident, init_db, list_incidents, save_incident, update_incident_review
from vision.config import ROOT, load_config
from vision.nudge import AudioNudge

app = FastAPI(
    title="CleanLoop API",
    description="AI Littering Prevention REST & WebSocket API",
    version="1.0.0"
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Mount evidence static files
storage_path = ROOT / "storage"
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")

# Mount frontend directory if built
frontend_path = ROOT / "frontend"
if frontend_path.exists() and (frontend_path / "index.html").exists():
    app.mount("/dashboard", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


# Connection Manager for WebSockets
class WebSocketHub:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


hub = WebSocketHub()


# Pydantic Schemas
class ReviewRequest(BaseModel):
    decision: str  # CONFIRMED_LITTER, FALSE_POSITIVE, CLEANED
    notes: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/")
def read_root():
    return {
        "app": "CleanLoop AI Litter Prevention API",
        "status": "online",
        "docs": "/docs",
        "dashboard": "/dashboard" if (frontend_path / "index.html").exists() else "/frontend/index.html"
    }


@app.get("/api/status")
def get_status():
    config = load_config()
    analytics = get_analytics()
    return {
        "status": "active",
        "camera": config["camera"],
        "analytics": analytics
    }


@app.get("/api/incidents")
def get_incidents(status: Optional[str] = "ALL"):
    return list_incidents(status)


@app.get("/api/incidents/{incident_id}")
def get_incident_by_id(incident_id: str):
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


@app.post("/api/incidents/{incident_id}/review")
def review_incident(incident_id: str, req: ReviewRequest):
    updated = update_incident_review(incident_id, req.decision, req.notes)
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found")
    return updated


@app.get("/api/analytics")
def read_analytics():
    return get_analytics()


@app.get("/api/leaderboard")
def read_leaderboard():
    from backend.database import get_station_leaderboard
    return get_station_leaderboard()


@app.post("/api/nudge/test")
def trigger_test_nudge():
    nudge = AudioNudge()
    nudge.play_nudge("CleanLoop system audio test. Please keep your surroundings clean.")
    nudge.stop()
    return {"status": "ok", "message": "Test audio nudge triggered"}


@app.post("/api/bmrcl/call_test")
def trigger_bmrcl_test_call():
    from vision.dispatch import BMRCLEscalator
    config = load_config()
    escalator = BMRCLEscalator(config)
    result = escalator.make_bmrcl_call(total_area=15500)
    return {
        "status": "ok",
        "dispatch": result
    }


@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await hub.connect(websocket)
    try:
        while True:
            # Keep connection open & handle incoming pings
            data = await websocket.receive_text()
            await websocket.send_json({"type": "PONG", "message": data})
    except WebSocketDisconnect:
        hub.disconnect(websocket)
