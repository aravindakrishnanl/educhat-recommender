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

# --- WORKSPACE SCHEMAS (Input) ---
class WorkspaceBaseGraph(BaseModel):
    model_type: Literal["GRAPH"]
    name: str
    branch: str 
    skills: str 
    interests: str

class WorkspaceBaseML(BaseModel):
    model_type: Literal["KMEANS", "SVD"]
    name: str
    branch: str 
    skills: str 
    interests: str
    projects_completed: Optional[str]
    certifications_completed: Optional[str]

WorkspaceCreate = Union[WorkspaceBaseGraph, WorkspaceBaseML]

# --- WORKSPACE SCHEMAS (Output) ---
class RecommendationResult(BaseModel):
    career_name: str
    score: float
    required_skills: List[str]
    recommended_courses: List[str]
    roadmap: str

class Workspace(BaseModel):
    id: int
    user_id: int
    model_type: str
    name: str
    branch: str
    skills: str
    interests: str
    projects_completed: Optional[str] = None
    certifications_completed: Optional[str] = None
    recommendations: List[RecommendationResult] | None = None
    created_at: datetime

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True

# --- NEW FEEDBACK SCHEMA ---
class FeedbackCreate(BaseModel):
    workspace_id: int
    career_name: str
    rating: Literal[0, 1]