from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=func.now())
    
    workspaces = relationship("Workspace", back_populates="owner", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")

class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    model_type = Column(String(10), nullable=False) 
    projects_completed = Column(Text, nullable=True)
    certifications_completed = Column(Text, nullable=True)
    
    name = Column(String)
    branch = Column(String)
    skills = Column(Text)       
    interests = Column(Text)    
    recommendations = Column(JSON, nullable=True) 
    created_at = Column(DateTime, default=func.now())

    owner = relationship("User", back_populates="workspaces")
    # NEW: Link to Feedback items for this workspace
    feedback_items = relationship("Feedback", back_populates="workspace", cascade="all, delete-orphan")

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # FIX: Added ON DELETE CASCADE to the Foreign Key constraint
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    
    career_name = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)
    
    created_at = Column(DateTime, default=func.now())
    
    user = relationship("User", back_populates="feedback")
    workspace = relationship("Workspace", back_populates="feedback_items")