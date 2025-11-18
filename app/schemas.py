from pydantic import BaseModel
from typing import List, Any, Literal, Optional, Union
from datetime import datetime

# --- AUTH SCHEMAS ---
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- WORKSPACE SCHEMAS ---

# Base schema for Graph Model (fewer inputs)
class WorkspaceBaseGraph(BaseModel):
    model_type: Literal["GRAPH"]
    name: str
    branch: str 
    skills: str 
    interests: str

# Base schema for K-Means/SVD Models (more inputs)
class WorkspaceBaseML(BaseModel):
    model_type: Literal["KMEANS", "SVD"]
    name: str
    branch: str 
    skills: str 
    interests: str
    projects_completed: str
    certifications_completed: str

# Union of all possible input schemas for the POST route
WorkspaceCreate = Union[WorkspaceBaseGraph, WorkspaceBaseML]

class RecommendationResult(BaseModel):
    career_name: str
    score: float
    required_skills: List[str]
    recommended_courses: List[str]
    roadmap: str

class Workspace(BaseModel):
    id: int
    user_id: int
    model_type: str # Store selected model type
    name: str
    branch: str
    skills: str
    interests: str
    # Projects/certs are optional in the database and output
    projects_completed: Optional[str] = None
    certifications_completed: Optional[str] = None
    recommendations: List[RecommendationResult] | None = None
    created_at: datetime

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True