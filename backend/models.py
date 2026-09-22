from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
import datetime

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    transcript = Column(String, index=True)
    gyeongsang = Column(Float)
    jeolla = Column(Float)
    chungcheong = Column(Float)
    standard = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
