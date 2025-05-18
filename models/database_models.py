from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text, JSON, DateTime, Float, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from utils.database import Base
from datetime import datetime
import enum

# Constants for action types
class ActionType:
    BATCH_GENERATED = "batch_generated"
    BATCH_DELETED = "batch_deleted"
    BATCH_NAME_EDITED = "batch_name_edited"
    TRANSACTION_EDITED = "transaction_edited"
    TRANSACTION_DELETED = "transaction_deleted"
    DISTRIBUTION_UPDATED = "distribution_updated"
    PERSONA_CREATED = "persona_created"
    PERSONA_UPDATED = "persona_updated"
    BATCH_DOWNLOADED_CSV = "batch_downloaded_csv"
    BATCH_DOWNLOADED_JSON = "batch_downloaded_json"
    BATCH_DOWNLOADED_EXCEL = "batch_downloaded_excel"

class AuditLog(Base):
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    action_type = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    entity_type = Column(String, nullable=False)  # 'batch', 'transaction', 'persona'
    entity_id = Column(String, nullable=False)    # ID of the affected entity
    details = Column(JSON, nullable=True)         # Additional context about the action
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    personas = relationship("Persona", back_populates="user")
    transaction_batches = relationship("TransactionBatch", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")


class Persona(Base):
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    description = Column(Text)
    config_json = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="personas")
    transaction_batches = relationship("TransactionBatch", back_populates="persona")


class TransactionBatch(Base):
    __tablename__ = "transaction_batches"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    persona_id = Column(Integer, ForeignKey("personas.id"))
    name = Column(String)
    months = Column(Integer, nullable=True)  # Store the selected number of months
    summary_json = Column(JSON, nullable=True)
    preview_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="transaction_batches")
    persona = relationship("Persona", back_populates="transaction_batches")
    transactions = relationship("Transaction", back_populates="batch")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("transaction_batches.id"))
    
    # Transaction fields
    transaction_id = Column(String, index=True)
    booking_date_time = Column(DateTime(timezone=True))
    value_date_time = Column(DateTime(timezone=True))
    amount = Column(Float)
    currency = Column(String)
    creditor_name = Column(String)
    creditor_account_iban = Column(String)
    debtor_name = Column(String)
    debtor_account_iban = Column(String)
    remittance_information_unstructured = Column(Text)
    category = Column(String)
    
    edited = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    batch = relationship("TransactionBatch", back_populates="transactions") 