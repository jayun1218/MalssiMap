from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./malssi.db"
# SQLAlchemy는 기본적으로 한 스레드에서 생성된 객체를 다른 스레드에서 사용하지 못하도록 방지합니다.
# FastAPI는 여러 스레드를 사용하므로 check_same_thread=False를 설정해야 합니다. (SQLite 한정)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
