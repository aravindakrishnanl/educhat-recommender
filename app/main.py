from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from typing import Annotated
from pathlib import Path
import os
from typing import List

# CORE PACKAGE IMPORTS
from .database import engine, Base, get_db
from . import models, recommender_core 

# Import specific schema classes explicitly (to break circular imports)
from .schemas import (
    User as UserSchema, 
    UserCreate, 
    Workspace, 
    WorkspaceCreate
)

# --- INITIAL SETUP ---
app = FastAPI(title="Career Recommender API")
# FIX: Using sha256_crypt for stable hashing across environments
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

    # 1. Run the recommendation logic
    recommendations_list = recommender_core.generate_recommendations(
        user_name=user.username,
        user_skills_str=workspace.skills,
        user_interests_str=workspace.interests
    )

    # 2. Store the workspace data and results
    db_workspace = models.Workspace(
        user_id=user_id,
        name=workspace.name,
        branch=workspace.branch,
        skills=workspace.skills,
        interests=workspace.interests,
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

# NEW FEATURE: Delete Workspace
@app.delete("/user/{user_id}/workspace/{workspace_id}")
def delete_workspace(user_id: int, workspace_id: int, db: db_dependency):
    # Find the workspace, ensuring it belongs to the authenticated user (user_id)
    workspace = db.query(models.Workspace).filter(
        models.Workspace.id == workspace_id,
        models.Workspace.user_id == user_id
    ).first()

    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found or unauthorized.")
    
    # Delete the record
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