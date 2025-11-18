from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# REPLACE with your actual PostgreSQL connection string
# Example: postgresql://user:password@host:port/dbname
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:admin@localhost/career_db" 

# Set up the engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a SessionLocal class for the database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for the ORM models
Base = declarative_base()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()