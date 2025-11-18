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
    
    # Using string reference "Workspace"
    workspaces = relationship("Workspace", back_populates="owner")

class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    branch = Column(String)
    skills = Column(Text)       
    interests = Column(Text)    
    recommendations = Column(JSON, nullable=True) 
    created_at = Column(DateTime, default=func.now())

    # Using string reference "User"
    owner = relationship("User", back_populates="workspaces")