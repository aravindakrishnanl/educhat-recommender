from pydantic import BaseModel
from typing import List, Any
from datetime import datetime

# --- AUTH SCHEMAS ---
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

# This is the class referenced as UserSchema in main.py
class User(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- WORKSPACE SCHEMAS ---
class WorkspaceBase(BaseModel):
    name: str
    branch: str 
    skills: str 
    interests: str

class WorkspaceCreate(WorkspaceBase):
    pass

class RecommendationResult(BaseModel):
    career_name: str
    score: float
    required_skills: List[str]
    recommended_courses: List[str]
    roadmap: str

class Workspace(WorkspaceBase):
    id: int
    user_id: int
    recommendations: List[RecommendationResult] | None = None
    created_at: datetime

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True