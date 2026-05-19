from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./audit_logs.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    amount = Column(Float)
    decision = Column(String)
    probability = Column(Float)
    confidence = Column(Float)
    investigator = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_prediction(amount, decision, prob, confidence, investigator="System"):
    db = SessionLocal()
    new_log = AuditLog(
        amount=amount,
        decision=decision,
        probability=prob,
        confidence=confidence,
        investigator=investigator
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    db.close()
    return new_log
