from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from backend.db.database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    role = Column(String)

class SystemEvent(Base):
    __tablename__ = "system_events"
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String)
    source = Column(String)
    message = Column(String)
    extra_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


class ModelPrediction(Base):
    """Auditable record for a centralized model decision or state transition."""
    __tablename__ = "model_predictions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    event_state = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
