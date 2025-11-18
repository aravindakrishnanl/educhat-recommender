from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from typing import Annotated, List
from pathlib import Path
import os

# CORE PACKAGE IMPORTS
from .database import engine, Base, get_db
# Import recommender_setup before recommender_core to ensure ML models are initialized
from . import models, recommender_setup 
from . import recommender_core # Contains the recommendation functions

# Import specific schema classes explicitly
from .schemas import (
    User as UserSchema, 
    UserCreate, 
    Workspace, 
    WorkspaceCreate,
    WorkspaceBaseGraph,
    WorkspaceBaseML
)

# --- INITIAL SETUP ---
app = FastAPI(title="Career Recommender API")
# Using sha256_crypt for stable hashing
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto") 
db_dependency = Annotated[Session, Depends(get_db)]

# Define the project base directory for robust path handling
BASE_DIR = Path(__file__).resolve().parent.parent

# Create tables on startup
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

# Serve static files (CSS) using an absolute path for reliability
app.mount(
    "/static", 
    StaticFiles(directory=os.path.join(BASE_DIR, "app/static")), 
    name="static"
)

# --- AUTH HELPERS ---

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password) 

# --- AUTH ROUTES ---

@app.post("/signup", response_model=UserSchema)
def signup(user_data: UserCreate, db: db_dependency):
    if db.query(models.User).filter(models.User.username == user_data.username).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    
    hashed_password = get_password_hash(user_data.password)
    db_user = models.User(username=user_data.username, hashed_password=hashed_password)
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/login")
def login(db: db_dependency, form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    
    return {"message": "Login successful", "user_id": user.id, "username": user.username}

# --- WORKSPACE ROUTES ---

@app.post("/user/{user_id}/workspace", response_model=Workspace)
def create_workspace(user_id: int, workspace: WorkspaceCreate, db: db_dependency):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    model_type = workspace.model_type
    recommendations_list = []

    # Prepare data based on the chosen model schema
    user_skills_str = workspace.skills
    user_interests_str = workspace.interests
    
    # Base data dictionary for ML models (SVD/KMEANS)
    ml_data = {
        'branch': workspace.branch,
        'skills': [s.strip() for s in user_skills_str.split(',') if s.strip()], # ML functions expect a list
        'interests': user_interests_str,
        'projects_completed': getattr(workspace, 'projects_completed', ""),
        'certifications_completed': getattr(workspace, 'certifications_completed', "")
    }

    # 1. Run the recommendation logic based on model type
    if model_type == "GRAPH":
        recommendations_list = recommender_core.recommend_graph_model(
            user_name=user.username,
            user_skills_str=user_skills_str,
            user_interests_str=user_interests_str
        )
    elif model_type == "SVD":
        recommendations_list = recommender_core.recommend_svd_model(ml_data)
    elif model_type == "KMEANS":
        recommendations_list = recommender_core.recommend_kmeans_model(ml_data)
    else:
        # Should be caught by Literal type hints in schemas, but safe to include
        raise HTTPException(status_code=400, detail="Invalid model type selected.")


    # 2. Store the workspace data and results
    db_workspace = models.Workspace(
        user_id=user_id,
        model_type=model_type,
        name=workspace.name,
        branch=workspace.branch,
        skills=user_skills_str,
        interests=user_interests_str,
        # Safely assign optional fields; they will be None if not provided by the schema/form
        projects_completed=getattr(workspace, 'projects_completed', None), 
        certifications_completed=getattr(workspace, 'certifications_completed', None),
        recommendations=recommendations_list 
    )

    db.add(db_workspace)
    db.commit()
    db.refresh(db_workspace)
    
    return Workspace.from_orm(db_workspace)

@app.get("/user/{user_id}/workspaces", response_model=List[Workspace])
def get_workspaces(user_id: int, db: db_dependency):
    workspaces = db.query(models.Workspace).filter(models.Workspace.user_id == user_id).all()
    return [Workspace.from_orm(ws) for ws in workspaces]

@app.delete("/user/{user_id}/workspace/{workspace_id}")
def delete_workspace(user_id: int, workspace_id: int, db: db_dependency):
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.user_id == user_id
    ).first()

    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found or unauthorized.")
    
    db.delete(workspace)
    db.commit()
    
    return {"message": f"Workspace ID {workspace_id} deleted successfully."}


# --- FRONTEND ROUTES (To serve HTML) ---

def get_template_response(template_name):
    """Utility function to read and serve HTML templates."""
    try:
        with open(BASE_DIR / "templates" / template_name, "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Template {template_name} not found. Check the 'templates' folder.")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return get_template_response("index.html")

@app.get("/auth", response_class=HTMLResponse)
async def get_auth():
    return get_template_response("auth.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    return get_template_response("dashboard.html")