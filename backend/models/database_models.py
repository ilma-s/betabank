from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, 
    Text, JSON, Boolean, Float, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum

Base = declarative_base()

class ActionType(str, Enum):
    BATCH_GENERATED = "batch_generated"
    BATCH_DELETED = "batch_deleted"
    TRANSACTION_DELETED = "transaction_deleted"
    TRANSACTION_EDITED = "transaction_edited"
    BATCH_NAME_EDITED = "batch_name_edited"
    BATCH_DOWNLOADED_CSV = "batch_downloaded_csv"
    BATCH_DOWNLOADED_JSON = "batch_downloaded_json"
    BATCH_DOWNLOADED_EXCEL = "batch_downloaded_excel"
    DISTRIBUTION_UPDATED = "distribution_updated"
    EXPLANATION_GENERATED = "explanation_generated"

class PatternType:
    TEMPORAL = "temporal"
    AMOUNT = "amount"
    CATEGORY = "category"
    DISTRIBUTION = "distribution"

class PatternLibrary(Base):
    __tablename__ = "pattern_library"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    pattern_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    rules = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TransactionExplanation(Base):
    __tablename__ = "transaction_explanations"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id", ondelete="CASCADE"), nullable=False)
    batch_id = Column(Integer, ForeignKey("transaction_batches.id", ondelete="CASCADE"), nullable=False)
    feature_importance = Column(JSON, nullable=False)
    applied_patterns = Column(JSON, nullable=True)
    explanation_text = Column(Text, nullable=False)
    confidence_score = Column(Float)
    meta_info = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    transaction = relationship("Transaction", back_populates="explanation", foreign_keys=[transaction_id])
    batch = relationship("TransactionBatch", back_populates="explanations")

class BatchExplanation(Base):
    __tablename__ = "batch_explanations"

    id = Column(Integer, primary_key=True)
    batch_id = Column(Integer, ForeignKey("transaction_batches.id", ondelete="CASCADE"), nullable=False)
    distribution_explanation = Column(JSON, nullable=False)
    temporal_patterns = Column(JSON, nullable=False)
    amount_patterns = Column(JSON, nullable=False)
    anomalies = Column(JSON, nullable=True)
    summary_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    batch = relationship("TransactionBatch", back_populates="explanation")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action_type = Column(SQLEnum(ActionType), nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="audit_logs")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    personas = relationship("Persona", back_populates="user")
    batches = relationship("TransactionBatch", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

class Persona(Base):
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    config_json = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="personas")
    batches = relationship("TransactionBatch", back_populates="persona")

class TransactionBatch(Base):
    __tablename__ = "transaction_batches"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=False)
    name = Column(String, nullable=False)
    preview_json = Column(JSON)
    months = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="batches")
    persona = relationship("Persona", back_populates="batches")
    transactions = relationship("Transaction", back_populates="batch", cascade="all, delete-orphan")
    explanation = relationship("BatchExplanation", back_populates="batch", uselist=False, cascade="all, delete-orphan")
    explanations = relationship("TransactionExplanation", back_populates="batch", cascade="all, delete-orphan")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    batch_id = Column(Integer, ForeignKey("transaction_batches.id", ondelete="CASCADE"), nullable=False)
    transaction_id = Column(String, nullable=False, unique=True)
    booking_date_time = Column(DateTime(timezone=True), nullable=False)
    value_date_time = Column(DateTime(timezone=True), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String, nullable=False)
    creditor_name = Column(String)
    creditor_account_iban = Column(String)
    debtor_name = Column(String)
    debtor_account_iban = Column(String)
    remittance_information_unstructured = Column(Text)
    category = Column(String)
    edited = Column(Boolean, default=False)

    # Relationships
    batch = relationship("TransactionBatch", back_populates="transactions")
    explanation = relationship("TransactionExplanation", back_populates="transaction", uselist=False, cascade="all, delete-orphan") 