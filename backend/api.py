from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import numpy as np
import logging
import torch
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.wgan_gp import WGAN_GP
from utils.data_processor import TransactionDataProcessor
from utils.evaluation_metrics import TransactionEvaluator
from utils.database import get_db
from models.database_models import (
    User, Persona, TransactionBatch, Transaction, 
    AuditLog, ActionType, TransactionExplanation, BatchExplanation
)
from models.explanation_service import ExplanationService

from passlib.context import CryptContext
from jose import JWTError, jwt

# Constants and Configuration
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# API Models
class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    password: str

class PersonaBase(BaseModel):
    name: str
    description: str

class PersonaCreate(PersonaBase):
    distribution: Optional[Dict[str, float]] = Field(None, description="Category distribution percentages")
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Custom transaction dataset")

class PersonaResponse(PersonaBase):
    id: int
    class Config:
        from_attributes = True

class TransactionAmount(BaseModel):
    amount: str
    currency: str

class TransactionCreate(BaseModel):
    bookingDateTime: str
    valueDateTime: str
    transactionAmount: TransactionAmount
    creditorName: str
    creditorAccount: Dict[str, str]
    debtorName: str
    debtorAccount: Dict[str, str]
    remittanceInformationUnstructured: str
    category: str

class TransactionUpdate(BaseModel):
    transactionAmount: Optional[TransactionAmount] = None
    category: Optional[str] = None
    creditorName: Optional[str] = None
    remittanceInformationUnstructured: Optional[str] = None
    useForTraining: Optional[bool] = Field(True, description="Whether to use this update for model training")

class TransactionResponse(BaseModel):
    transactionId: str
    bookingDateTime: str
    valueDateTime: str
    transactionAmount: TransactionAmount
    creditorName: str
    creditorAccount: Dict[str, str]
    debtorName: str
    debtorAccount: Dict[str, str]
    remittanceInformationUnstructured: str
    category: str
    edited: bool = False

class BatchCreate(BaseModel):
    name: Optional[str] = None
    batch_size: int = Field(100, description="Number of transactions to generate (must be a multiple of 100)")

class BatchResponse(BaseModel):
    id: int
    name: str
    persona_id: int
    persona_name: str
    created_at: str
    transaction_count: int
    preview: Dict[str, Any]
    transactions: Optional[List[TransactionResponse]] = None
    months: int

class DistributionUpdate(BaseModel):
    distribution: Dict[str, float] = Field(..., description="Category distribution percentages, must sum to 1.0")
    useForTraining: bool = Field(True, description="Whether to use this distribution for model training")
    batchId: Optional[int] = Field(None, description="Batch ID to regenerate with new distribution")

class BatchUpdate(BaseModel):
    name: str

# API Documentation
description = """
🏦 BetaBank API

Generate and manage synthetic banking transactions with AI-powered personas.

## Features

* 🔐 JWT Authentication
* 👤 Persona Management
* 💰 Transaction Generation
* 📊 Distribution Control
* 📈 Model Training
* 📁 Batch Management

## Getting Started

1. Register a new user account
2. Get an access token
3. Create or select a persona
4. Generate transactions
5. Manage and analyze your transaction batches

"""

tags_metadata = [
    {
        "name": "authentication",
        "description": "User authentication and registration operations",
    },
    {
        "name": "personas",
        "description": "Manage transaction generation personas and their distributions",
    },
    {
        "name": "transactions",
        "description": "Generate and manage synthetic transactions",
    },
    {
        "name": "batches",
        "description": "Work with transaction batches - create, update, delete, and download batches",
    },
    {
        "name": "explanations",
        "description": "Get AI-generated explanations for batches and individual transactions",
    },
    {
        "name": "audit",
        "description": "Access audit logs and system activity",
    }
]

app = FastAPI(
    title="BetaBank API",
    description=description,
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "BetaBank Support",
        "email": "support@betabank.example.com",
    },
    license_info={
        "name": "Private License",
        "url": "https://example.com/license",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

PERSONAS = {
    "crypto_enthusiast": {"name": "Crypto Enthusiast", "description": "A tech-savvy individual who primarily invests in cryptocurrencies", "dataset_path": "testing_datasets/crypto_enthusiast.json"},
    "shopping_addict": {"name": "Shopping Addict", "description": "Someone who frequently shops online and at retail stores", "dataset_path": "testing_datasets/shopping_addict.json"},
    "gambling_addict": {"name": "Gambling Addict", "description": "A person with a regular gambling habit", "dataset_path": "testing_datasets/gambling_addict.json"},
    "money_mule": {"name": "Money Mule", "description": "Individual involved in moving illegally acquired money", "dataset_path": "testing_datasets/money_mule.json"}
}

models = {}
data_processor = TransactionDataProcessor()

# Global evaluator instance
evaluator = None

def verify_password(plain_password, hashed_password):
    logger.debug(f"Verifying password. Plain: {plain_password}, Hash: {hashed_password}")
    result = pwd_context.verify(plain_password, hashed_password)
    logger.debug(f"Password verification result: {result}")
    return result

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

@app.post("/token", response_model=Token, tags=["authentication"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Get an access token for API authentication.

    - **username**: Your username
    - **password**: Your password
    
    Returns a JWT token valid for 30 minutes.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", tags=["authentication"])
async def register_user(
    user: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.

    - **username**: Desired username
    - **password**: Secure password
    
    Creates a new user account and initializes default personas.
    """
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, password_hash=hashed_password)
    db.add(new_user)
    db.flush()  # Flush to get the user ID
    
    # Update persona configurations to use S3 paths
    s3_personas = {
        "crypto_enthusiast": {
            "name": "Crypto Enthusiast",
            "description": "A tech-savvy individual who primarily invests in cryptocurrencies",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/crypto_enthusiast.json"
        },
        "shopping_addict": {
            "name": "Shopping Addict",
            "description": "Someone who frequently shops online and at retail stores",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/shopping_addict.json"
        },
        "gambling_addict": {
            "name": "Gambling Addict",
            "description": "A person with a regular gambling habit",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/gambling_addict.json"
        },
        "money_mule": {
            "name": "Money Mule",
            "description": "Individual involved in moving illegally acquired money",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/money_mule.json"
        }
    }
    
    for persona_id, persona_data in s3_personas.items():
        persona = Persona(
            user_id=new_user.id,
            name=persona_data["name"],
            description=persona_data["description"],
            config_json={"dataset_path": persona_data["dataset_path"]}
        )
        db.add(persona)
    
    db.commit()
    return {"message": "User registered successfully"}

@app.post("/ensure-personas", tags=["personas"])
async def ensure_personas(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ensure all default personas exist for the user.
    
    Creates any missing personas and updates existing ones with correct configurations.
    """
    existing_personas = db.query(Persona).filter(Persona.user_id == user.id).all()
    existing_names = {p.name.lower() for p in existing_personas}
    
    s3_personas = {
        "crypto_enthusiast": {
            "name": "Crypto Enthusiast",
            "description": "A tech-savvy individual who primarily invests in cryptocurrencies",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/crypto_enthusiast.json"
        },
        "shopping_addict": {
            "name": "Shopping Addict",
            "description": "Someone who frequently shops online and at retail stores",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/shopping_addict.json"
        },
        "gambling_addict": {
            "name": "Gambling Addict",
            "description": "A person with a regular gambling habit",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/gambling_addict.json"
        },
        "money_mule": {
            "name": "Money Mule",
            "description": "Individual involved in moving illegally acquired money",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/money_mule.json"
        }
    }
    
    added = []
    for persona_id, data in s3_personas.items():
        if data["name"].lower() not in existing_names:
            new_persona = Persona(
                user_id=user.id,
                name=data["name"],
                description=data["description"],
                config_json={"dataset_path": data["dataset_path"]}
            )
            db.add(new_persona)
            added.append(data["name"])
    
    if added:
        db.commit()
        return {"message": f"Added missing personas: {', '.join(added)}"}
    
    # Update existing personas with correct S3 config
    updated = []
    for persona in existing_personas:
        for persona_id, data in s3_personas.items():
            if persona.name == data["name"]:
                if persona.config_json.get("dataset_path") != data["dataset_path"]:
                    persona.config_json = {"dataset_path": data["dataset_path"]}
                    updated.append(persona.name)
                break
    
    if updated:
        db.commit()
        return {"message": f"Updated personas with S3 paths: {', '.join(updated)}"}
    
    return {"message": "All personas exist and are up to date"}

@app.get("/personas", response_model=List[PersonaResponse], tags=["personas"])
async def get_personas(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all available personas for the authenticated user.
    
    Returns a list of personas with their configurations and descriptions.
    """
    personas = db.query(Persona).filter(Persona.user_id == user.id).all()
    return personas

@app.post("/create-persona", tags=["personas"])
async def create_persona(
    persona_data: PersonaCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new custom persona with specified distribution.
    
    Creates a persona with custom category distribution for transaction generation.
    """
    # Check if persona name already exists for this user
    existing_persona = db.query(Persona).filter(
        Persona.user_id == user.id,
        Persona.name == persona_data.name
    ).first()
    
    if existing_persona:
        raise HTTPException(status_code=400, detail="Persona with this name already exists")
    
    # Validate distribution if provided
    if persona_data.distribution:
        total = sum(persona_data.distribution.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise HTTPException(
                status_code=400, 
                detail="Distribution values must sum to 1.0"
            )
    
    # Create the persona
    new_persona = Persona(
        user_id=user.id,
        name=persona_data.name,
        description=persona_data.description,
        config_json={
            "distribution": persona_data.distribution,
            "dataset": persona_data.dataset,
            "is_custom": True
        }
    )
    
    db.add(new_persona)
    db.commit()
    db.refresh(new_persona)
    
    return {
        "id": new_persona.id,
        "message": f"Persona '{persona_data.name}' created successfully"
    }

@app.get("/batches", response_model=List[BatchResponse], tags=["batches"])
async def get_batches(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all transaction batches for the authenticated user.
    
    Returns a list of batches with their basic information and preview data.
    """
    batches = (
        db.query(TransactionBatch)
        .filter(TransactionBatch.user_id == user.id)
        .order_by(TransactionBatch.created_at.desc())
        .all()
    )

    result = []
    for batch in batches:
        persona = db.query(Persona).filter(Persona.id == batch.persona_id).first()
        result.append({
            "id": batch.id,
            "name": batch.name,
            "persona_id": batch.persona_id,
            "persona_name": persona.name if persona else "Unknown",
            "created_at": batch.created_at.isoformat(),
            "transaction_count": len(batch.transactions) if batch.transactions else 0,
            "preview": batch.preview_json,
            "months": batch.months
        })

    return result  # Return the list directly instead of wrapping it in a dictionary

@app.get("/batches/evaluation-metrics", tags=["batches"])
async def get_all_batch_evaluation_metrics(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get evaluation metrics for all generated batches for the current user.
    """
    logs = db.query(AuditLog).filter(
        AuditLog.user_id == user.id,
        AuditLog.action_type == ActionType.BATCH_GENERATED
    ).order_by(AuditLog.timestamp.desc()).all()

    results = []
    for log in logs:
        details = log.details or {}
        metrics = details.get("evaluation_metrics")
        if metrics:
            results.append({
                "batch_id": log.entity_id,
                "persona_id": details.get("persona_id"),
                "months": details.get("months"),
                "transaction_count": details.get("transaction_count"),
                "timestamp": log.timestamp.isoformat(),
                **metrics
            })
    return {"metrics": results}

@app.get("/batches/{batch_id}", response_model=BatchResponse, tags=["batches"])
async def get_batch(
    batch_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific batch.
    
    Returns the batch with all its transactions and metadata.
    """
    batch = (
        db.query(TransactionBatch)
        .filter(TransactionBatch.id == batch_id, TransactionBatch.user_id == user.id)
        .first()
    )
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Get the persona name
    persona = db.query(Persona).filter(Persona.id == batch.persona_id).first()
    
    # Get all transactions for this batch
    transactions = (
        db.query(Transaction)
        .filter(Transaction.batch_id == batch_id)
        .order_by(Transaction.booking_date_time)
        .all()
    )
    
    # Convert transactions to the expected format
    formatted_transactions = []
    for tx in transactions:
        formatted_transactions.append({
            "transactionId": tx.transaction_id,
            "bookingDateTime": tx.booking_date_time.isoformat(),
            "valueDateTime": tx.value_date_time.isoformat(),
            "transactionAmount": {
                "amount": f"{tx.amount:.2f}",
                "currency": tx.currency
            },
            "creditorName": tx.creditor_name,
            "creditorAccount": {
                "iban": tx.creditor_account_iban
            },
            "debtorName": tx.debtor_name,
            "debtorAccount": {
                "iban": tx.debtor_account_iban
            },
            "remittanceInformationUnstructured": tx.remittance_information_unstructured,
            "category": tx.category
        })
    
    return {
        "id": batch.id,
        "name": batch.name,
        "persona_id": batch.persona_id,
        "persona_name": persona.name if persona else "Unknown",
        "created_at": batch.created_at.isoformat(),
        "transaction_count": len(transactions),
        "preview": batch.preview_json,
        "transactions": formatted_transactions,
        "months": batch.months
    }

@app.post("/generate/{persona_id}", response_model=BatchResponse, tags=["transactions"])
async def generate_transactions(
    persona_id: int,
    batch_config: BatchCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate synthetic transactions for a specific persona.
    
    Creates a new batch of transactions based on the persona's configuration.
    """
    persona = db.query(Persona).filter(Persona.id == persona_id, Persona.user_id == user.id).first()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    dataset_path = persona.config_json.get("dataset_path")
    if not dataset_path:
        # For distribution-only personas, use a default dataset based on the categories
        if persona.config_json.get("custom_distribution"):
            # Use shopping_addict as default base dataset since it has a good variety of transaction types
            dataset_path = "s3://synthetic-personas-training-datasets/testing_datasets/shopping_addict.json"
            logger.info(f"Using default dataset for distribution-only persona: {dataset_path}")
        else:
            raise HTTPException(status_code=500, detail="No dataset path configured")
    
    try:
        # Get or create model key based on dataset path and persona
        model_key = f"persona_{persona_id}_{hash(dataset_path)}"
        model = models.get(model_key)
        
        if not model:
            X, C = data_processor.load_data(dataset_path, persona.name)
            input_dim = X.shape[1]  # Number of numerical features
            condition_dim = C.shape[1]  # Number of categories
            tensor_X, tensor_C = torch.FloatTensor(X), torch.FloatTensor(C)
            
            # Create model with optimized parameters
            # Note: Generator expects input_dim + condition_dim for the first layer
            model = WGAN_GP(
                input_dim=input_dim,  # Numerical features
                output_dim=input_dim,  # Output same size as input
                condition_dim=condition_dim,  # Number of categories
                device=data_processor.device
            )
            
            # Minimal training for speed
            n_epochs = 15  # Further reduced from 25
            batch_size = 256  # Increased from 128
            n_batches = len(X) // batch_size
            
            # Train with larger batches and fewer iterations
            for epoch in range(n_epochs):
                total_loss = 0
                for i in range(0, len(X), batch_size):
                    batch_X = tensor_X[i:i + batch_size]
                    batch_C = tensor_C[i:i + batch_size]
                    stats = model.train_step(batch_X, batch_C)
                    total_loss += stats['g_loss']
                
                if epoch % 5 == 0:
                    avg_loss = total_loss / n_batches
                    logger.info(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")
            
            models[model_key] = model

        # Generate data
        n_samples = batch_config.batch_size
        if n_samples % 100 != 0:
            raise HTTPException(status_code=400, detail="Batch size must be a multiple of 100.")
        categories = data_processor.category_columns
        
        # Get target distribution for the persona
        target_distribution = None
        if persona.config_json.get("custom_distribution"):
            # Use custom distribution if available
            target_distribution = {
                cat: {"min": pct, "max": pct}
                for cat, pct in persona.config_json["custom_distribution"].items()
            }
        else:
            # Fall back to default distribution
            target_distribution = data_processor.ensure_category_distribution(persona.name)
        
        if not target_distribution:
            chosen_categories = random.choices(list(categories), k=n_samples)
        else:
            category_weights = {
                cat: (rng["min"] + rng["max"]) / 2 for cat, rng in target_distribution.items()
            }
            total_weight = sum(category_weights.values())
            category_probs = {cat: w / total_weight for cat, w in category_weights.items()}
            # Deterministic assignment for exact proportions
            cats = list(category_probs.keys())
            probs = list(category_probs.values())
            counts = [int(round(p * n_samples)) for p in probs]
            # Adjust for rounding errors
            while sum(counts) < n_samples:
                counts[counts.index(max(counts))] += 1
            while sum(counts) > n_samples:
                counts[counts.index(max(counts))] -= 1
            chosen_categories = []
            for cat, count in zip(cats, counts):
                chosen_categories.extend([cat] * count)
            np.random.shuffle(chosen_categories)

        # Generate in larger batches
        batch_size = 100  # Increased from 50
        n_batches = (n_samples + batch_size - 1) // batch_size
        all_generated_data = []
        
        # Pre-compute category indices for faster lookup
        category_indices = {}
        for cat in set(chosen_categories):
            # Handle both prefixed and unprefixed category names
            if cat.startswith('category_'):
                category_name = cat
            else:
                category_name = f"category_{cat}"
            
            try:
                category_indices[cat] = list(categories).index(category_name)
            except ValueError:
                # If not found with prefix, try without prefix
                try:
                    category_indices[cat] = list(categories).index(cat)
                except ValueError:
                    logger.error(f"Could not find category index for {cat}")
                    raise ValueError(f"Invalid category: {cat}")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_categories = chosen_categories[start_idx:end_idx]
            
            condition_matrix = np.zeros((len(batch_categories), len(categories)))
            for j, cat in enumerate(batch_categories):
                idx = category_indices.get(cat)
                if idx is not None:
                    condition_matrix[j, idx] = 1
                else:
                    condition_matrix[j, random.randrange(len(categories))] = 1
            
            batch_generated = model.generate(len(batch_categories), condition_matrix)
            batch_combined = np.hstack([batch_generated, condition_matrix])
            all_generated_data.append(batch_combined)
        
        combined_data = np.vstack(all_generated_data)
        transactions = data_processor.inverse_transform(combined_data)
        
        # Bulk insert transactions
        default_name = f"Batch {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        months = batch_config.batch_size // 100
        batch = TransactionBatch(
            user_id=user.id,
            persona_id=persona_id,
            name=batch_config.name if batch_config.name else default_name,
            preview_json={"count": len(transactions)},
            months=months  # Deprecated, kept for backward compatibility
        )
        db.add(batch)
        db.flush()
        
        # Prepare all transactions at once
        db_transactions = []
        for tx in transactions:
            # Parse the date string - handle both formats
            try:
                booking_date = datetime.strptime(tx["bookingDateTime"], "%d/%m/%Y %H:%M:%S")
            except ValueError:
                try:
                    booking_date = datetime.fromisoformat(tx["bookingDateTime"])
                except ValueError:
                    logger.error(f"Invalid date format: {tx['bookingDateTime']}")
                    raise HTTPException(status_code=500, detail=f"Invalid date format: {tx['bookingDateTime']}")
            
            value_date = booking_date  # Use same date for value_date
            
            db_transactions.append(
                Transaction(
                    batch_id=batch.id,
                    transaction_id=tx["transactionId"],
                    booking_date_time=booking_date,
                    value_date_time=value_date,
                    amount=float(tx["transactionAmount"]["amount"]),
                    currency=tx["transactionAmount"]["currency"],
                    creditor_name=tx["creditorName"],
                    creditor_account_iban=tx["creditorAccount"]["iban"],
                    debtor_name=tx["debtorName"],
                    debtor_account_iban=tx["debtorAccount"]["iban"],
                    remittance_information_unstructured=tx["remittanceInformationUnstructured"],
                    category=tx["category"]
                )
            )
        
        # Bulk insert all transactions
        db.bulk_save_objects(db_transactions)
        db.commit()
        
        # Generate feature importances and explanations
        feature_importances = []
        for tx in transactions:
            tx_tensor, condition_tensor = data_processor.transaction_to_tensor(tx)
            with torch.no_grad():
                _ = model.generator(tx_tensor, condition_tensor)
                importance = model.generator.get_feature_importance()
                # Convert importance tensor to dictionary format
                importance_dict = {
                    'amount': float(importance[0][0]),
                    'day_of_month': float(importance[0][1])
                }
                # Add category importances
                for i, cat in enumerate(data_processor.category_columns):
                    category_name = cat.replace('category_', '')
                    importance_dict[category_name] = float(importance[0][i + 2])  # Start at index 2 since we have 2 numerical features
                feature_importances.append(importance_dict)
        
        # Generate explanations using the explanation service
        explanation_service = ExplanationService(db)
        explanation_service.process_batch(batch, feature_importances)
        
        # Calculate evaluation metrics
        try:
            global evaluator
            if evaluator is None:
                evaluator = TransactionEvaluator(device=data_processor.device)
            
            # Load real training data for comparison
            real_transactions = []
            try:
                X, C = data_processor.load_data(dataset_path, persona.name)
                # Convert back to transaction format for evaluation
                real_data = data_processor.inverse_transform(np.hstack([X, C]))
                # Use larger subset for better evaluation quality
                real_transactions = real_data[:min(len(real_data), 2000)]  # Increased from 1000
                
                # Ensure we have enough diverse data for evaluation
                if len(real_transactions) < 500:
                    logger.warning(f"Limited real data available for evaluation: {len(real_transactions)} samples")
                
                logger.info(f"Loaded {len(real_transactions)} real transactions for evaluation")
            except Exception as e:
                logger.warning(f"Could not load real data for evaluation: {e}")
                real_transactions = []
            
            # Evaluate the generated batch
            metrics = evaluator.evaluate_batch(real_transactions, transactions)
            
            # Display metrics in console
            print("\n" + "="*60)
            print("🎯 BATCH GENERATION EVALUATION METRICS")
            print("="*60)
            print(f"📊 Batch ID: {batch.id}")
            print(f"👤 Persona: {persona.name}")
            print(f"📅 Generated: {len(transactions)} transactions over {months} months")
            print("-" * 60)
            print(f"🏆 Inception Score: {metrics['inception_score']:.4f}")
            print(f"📏 Fréchet Inception Distance (FID): {metrics['fid_score']:.4f}")
            print(f"🎲 Diversity Score: {metrics['diversity_score']:.4f}")
            print(f"🎭 Realism Score: {metrics['realism_score']:.4f}")
            print(f"⭐ Overall Quality Score: {metrics['overall_score']:.4f}")
            print("-" * 60)
            
            # Quality assessment
            if metrics['inception_score'] > 2.0:
                inception_quality = "🟢 EXCELLENT"
            elif metrics['inception_score'] > 1.5:
                inception_quality = "🟡 GOOD"
            else:
                inception_quality = "🔴 NEEDS IMPROVEMENT"
            
            if metrics['fid_score'] < 50:
                fid_quality = "🟢 EXCELLENT"
            elif metrics['fid_score'] < 100:
                fid_quality = "🟡 GOOD"
            else:
                fid_quality = "🔴 NEEDS IMPROVEMENT"
            
            print(f"Inception Score Quality: {inception_quality}")
            print(f"FID Quality: {fid_quality}")
            print("="*60)
            print()
            
            # Store metrics in audit log
            # Convert numpy values to Python floats for JSON serialization
            serializable_metrics = {
                "inception_score": float(metrics['inception_score']),
                "fid_score": float(metrics['fid_score']),
                "diversity_score": float(metrics['diversity_score']),
                "realism_score": float(metrics['realism_score']),
                "overall_score": float(metrics['overall_score'])
            }
            audit_details = {
                "persona_id": persona_id,
                "months": months,
                "transaction_count": len(transactions),
                "explanations_generated": True,
                "evaluation_metrics": serializable_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {e}")
            audit_details = {
                "persona_id": persona_id,
                "months": months,
                "transaction_count": len(transactions),
                "explanations_generated": True,
                "evaluation_error": str(e)
            }
        
        # Create audit log for batch generation
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.BATCH_GENERATED,
            entity_type="batch",
            entity_id=str(batch.id),
            details=audit_details
        )
        db.add(audit_log)
        db.commit()
        
        return {
            "id": batch.id,
            "name": batch.name,
            "persona_id": batch.persona_id,
            "persona_name": persona.name,
            "created_at": batch.created_at.isoformat(),
            "transaction_count": len(transactions),
            "preview": batch.preview_json,
            "transactions": transactions,
            "months": months
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error generating transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating transactions: {str(e)}")

@app.post("/update-personas", tags=["personas"])
async def update_personas(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update existing personas with correct configurations.
    """
    updated = []
    for persona in user.personas:
        for persona_id, data in PERSONAS.items():
            if persona.name == data["name"]:
                persona.config_json = {"dataset_path": data["dataset_path"]}
                updated.append(persona.name)
                break
    
    if updated:
        db.commit()
        return {"message": f"Updated personas: {', '.join(updated)}"}
    
    return {"message": "No personas needed updating"}

@app.delete("/batches/{batch_id}", tags=["batches"])
async def delete_batch(
    batch_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a specific batch and all its transactions.
    
    This operation cannot be undone.
    """
    # Find the batch and verify ownership
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    try:
        # Create audit log before deletion
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.BATCH_DELETED,
            entity_type="batch",
            entity_id=str(batch_id),
            details={
                "batch_name": batch.name,
                "persona_id": batch.persona_id,
                "transaction_count": len(batch.transactions)
            }
        )
        db.add(audit_log)
        
        # Delete associated transactions first
        db.query(Transaction).filter(Transaction.batch_id == batch_id).delete()
        
        # Delete the batch
        db.delete(batch)
        db.commit()
        
        return {"message": "Batch deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting batch: {str(e)}")

@app.patch("/batches/{batch_id}", tags=["batches"])
async def update_batch(
    batch_id: int,
    batch_update: BatchUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a batch's metadata (e.g., name).
    """
    # Find the batch and verify ownership
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    try:
        old_name = batch.name
        # Update the batch name
        batch.name = batch_update.name
        
        # Create audit log for the name change
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.BATCH_NAME_EDITED,
            entity_type="batch",
            entity_id=str(batch_id),
            details={
                "old_name": old_name,
                "new_name": batch_update.name
            }
        )
        db.add(audit_log)
        db.commit()
        
        return {"message": "Batch updated successfully", "name": batch_update.name}
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating batch: {str(e)}")

@app.get("/batches/{batch_id}/download/{format}", tags=["batches"])
async def download_batch(
    batch_id: int,
    format: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Download a batch in the specified format.
    
    Available formats:
    - csv: Comma-separated values
    - json: JSON array of transactions
    - excel: Microsoft Excel spreadsheet
    """
    if format not in ["csv", "json", "excel"]:
        raise HTTPException(status_code=400, detail="Invalid format. Must be csv, json, or excel")
    
    # Find the batch and verify ownership
    batch = (
        db.query(TransactionBatch)
        .filter(TransactionBatch.id == batch_id, TransactionBatch.user_id == user.id)
        .first()
    )
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    try:
        # Get all transactions for this batch
        transactions = (
            db.query(Transaction)
            .filter(Transaction.batch_id == batch_id)
            .order_by(Transaction.booking_date_time)
            .all()
        )
        
        # Format transactions for export
        formatted_transactions = [{
            "transaction_id": tx.transaction_id,
            "booking_date_time": tx.booking_date_time.isoformat(),
            "value_date_time": tx.value_date_time.isoformat(),
            "amount": f"{tx.amount:.2f}",
            "currency": tx.currency,
            "creditor_name": tx.creditor_name,
            "creditor_account_iban": tx.creditor_account_iban,
            "debtor_name": tx.debtor_name,
            "debtor_account_iban": tx.debtor_account_iban,
            "description": tx.remittance_information_unstructured,
            "category": tx.category,
            "edited": tx.edited
        } for tx in transactions]
        
        # Create audit log for the download
        action_type = getattr(ActionType, f"BATCH_DOWNLOADED_{format.upper()}")
        audit_log = AuditLog(
            user_id=user.id,
            action_type=action_type,
            entity_type="batch",
            entity_id=str(batch_id),
            details={
                "format": format,
                "transaction_count": len(transactions),
                "batch_name": batch.name
            }
        )
        db.add(audit_log)
        db.commit()
        
        # Generate the appropriate format
        if format == "json":
            return formatted_transactions
        elif format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=formatted_transactions[0].keys())
            writer.writeheader()
            writer.writerows(formatted_transactions)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="{batch.name}_{datetime.now().strftime("%Y%m%d")}.csv"'
                }
            )
        else:  # excel
            import pandas as pd
            from io import BytesIO
            
            df = pd.DataFrame(formatted_transactions)
            output = BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            
            return Response(
                content=output.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f'attachment; filename="{batch.name}_{datetime.now().strftime("%Y%m%d")}.xlsx"'
                }
            )
            
    except Exception as e:
        db.rollback()
        logger.error(f"Error downloading batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading batch: {str(e)}")

@app.patch("/transactions/{transaction_id}", response_model=TransactionResponse, tags=["transactions"])
async def update_transaction(
    transaction_id: str,
    transaction_update: TransactionUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a specific transaction's details.
    
    Optionally use the update for model training.
    """
    # Find the transaction and verify ownership through batch
    transaction = (
        db.query(Transaction)
        .join(TransactionBatch)
        .filter(
            Transaction.transaction_id == transaction_id,
            TransactionBatch.user_id == user.id
        )
        .first()
    )
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    try:
        # Store original values for audit log
        original_values = {
            "amount": transaction.amount,
            "category": transaction.category,
            "description": transaction.remittance_information_unstructured,
            "creditor_name": transaction.creditor_name
        }
        
        # Update allowed fields from the TransactionUpdate payload
        if transaction_update.transactionAmount is not None:
            transaction.amount = float(transaction_update.transactionAmount.amount)
            transaction.currency = transaction_update.transactionAmount.currency
        if transaction_update.remittanceInformationUnstructured is not None:
            transaction.remittance_information_unstructured = transaction_update.remittanceInformationUnstructured
        if transaction_update.category is not None:
            transaction.category = transaction_update.category
        if transaction_update.creditorName is not None:
            transaction.creditor_name = transaction_update.creditorName
        transaction.edited = True
        
        # Get the useForTraining flag
        use_for_training = transaction_update.useForTraining if transaction_update.useForTraining is not None else True
        
        # Create audit log for the edit
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.TRANSACTION_EDITED,
            entity_type="transaction",
            entity_id=transaction_id,
            details={
                "batch_id": transaction.batch_id,
                "original_values": original_values,
                "new_values": {
                    "amount": transaction.amount,
                    "category": transaction.category,
                    "description": transaction.remittance_information_unstructured,
                    "creditor_name": transaction.creditor_name
                },
                "use_for_training": use_for_training
            }
        )
        db.add(audit_log)
        
        # If useForTraining is True, update the training data
        if use_for_training:
            # Add the transaction to the training dataset
            # Assuming TrainingData model exists and is imported
            # from models.database_models import TrainingData
            # training_data = TrainingData(
            #     transaction_id=transaction_id,
            #     batch_id=transaction.batch_id,
            #     user_id=user.id,
            #     data_type="transaction_edit",
            #     data={
            #         "amount": transaction.amount,
            #         "category": transaction.category,
            #         "description": transaction.remittance_information_unstructured,
            #         "creditor_name": transaction.creditor_name,
            #         "original_values": original_values
            #     }
            # )
            # db.add(training_data)
            pass # Placeholder for training data update if implemented
        
        db.commit()
        
        # Return the updated transaction as a TransactionResponse
        return {
            "transactionId": transaction.transaction_id,
            "bookingDateTime": transaction.booking_date_time.isoformat(),
            "valueDateTime": transaction.value_date_time.isoformat(),
            "transactionAmount": {
                "amount": f"{transaction.amount:.2f}",
                "currency": transaction.currency
            },
            "creditorName": transaction.creditor_name,
            "creditorAccount": {
                "iban": transaction.creditor_account_iban
            },
            "debtorName": transaction.debtor_name,
            "debtorAccount": {
                "iban": transaction.debtor_account_iban
            },
            "remittanceInformationUnstructured": transaction.remittance_information_unstructured,
            "category": transaction.category,
            "edited": transaction.edited
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating transaction: {str(e)}")

@app.delete("/transactions/{transaction_id}", tags=["transactions"])
async def delete_transaction(
    transaction_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a specific transaction.
    
    This operation cannot be undone.
    """
    # Find the transaction and verify ownership through batch
    transaction = (
        db.query(Transaction)
        .join(TransactionBatch)
        .filter(
            Transaction.transaction_id == transaction_id,
            TransactionBatch.user_id == user.id
        )
        .first()
    )
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    try:
        # Create audit log before deletion
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.TRANSACTION_DELETED,
            entity_type="transaction",
            entity_id=transaction_id,
            details={
                "batch_id": transaction.batch_id,
                "amount": transaction.amount,
                "category": transaction.category,
                "description": transaction.remittance_information_unstructured,
                "creditor_name": transaction.creditor_name
            }
        )
        db.add(audit_log)
        
        # Delete the transaction
        db.delete(transaction)
        db.commit()
        
        return {"message": "Transaction deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting transaction: {str(e)}")

@app.get("/audit-logs", tags=["audit"])
async def get_audit_logs(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
    action_type: Optional[str] = None,
    entity_type: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    """
    Get audit logs for user actions.
    
    Filter by:
    - action_type: Type of action performed
    - entity_type: Type of entity affected
    - date range: from_date and to_date
    """
    query = db.query(AuditLog).filter(AuditLog.user_id == user.id)
    
    if action_type:
        query = query.filter(AuditLog.action_type == ActionType[action_type])
    if entity_type:
        query = query.filter(AuditLog.entity_type == entity_type)
    if from_date:
        query = query.filter(AuditLog.timestamp >= datetime.fromisoformat(from_date))
    if to_date:
        query = query.filter(AuditLog.timestamp <= datetime.fromisoformat(to_date))
    
    total = query.count()
    logs = query.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()
    
    return {
        "total": total,
        "logs": [{
            "id": log.id,
            "action_type": log.action_type.value,
            "timestamp": log.timestamp.isoformat(),
            "entity_type": log.entity_type,
            "entity_id": log.entity_id,
            "details": log.details
        } for log in logs]
    }

@app.get("/batches/{batch_id}/explanation", tags=["explanations"])
async def get_batch_explanation(
    batch_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get AI-generated explanation for a batch of transactions.
    
    Returns:
    - Distribution patterns across transaction categories
    - Temporal patterns in transaction timing
    - Amount patterns and trends
    - Detected anomalies
    - Overall batch summary
    """
    # Find the batch and verify ownership
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    explanation = db.query(BatchExplanation).filter(
        BatchExplanation.batch_id == batch_id
    ).first()
    
    if not explanation:
        raise HTTPException(status_code=404, detail="No explanation found for this batch")
    
    return {
        "batch_id": batch_id,
        "distribution_explanation": explanation.distribution_explanation,
        "temporal_patterns": explanation.temporal_patterns,
        "amount_patterns": explanation.amount_patterns,
        "anomalies": explanation.anomalies,
        "summary_text": explanation.summary_text
    }

@app.patch("/personas/{persona_id}/distribution", tags=["personas"])
async def update_persona_distribution(
    persona_id: int,
    distribution_update: DistributionUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update persona distribution and optionally regenerate batch with new distribution.
    
    - **distribution**: Category distribution percentages (must sum to 1.0)
    - **useForTraining**: Whether to use this distribution for model training
    - **batchId**: Optional batch ID to regenerate with new distribution
    """
    # Find the persona and verify ownership
    persona = db.query(Persona).filter(
        Persona.id == persona_id,
        Persona.user_id == user.id
    ).first()
    
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    try:
        # Validate distribution
        data_processor.validate_custom_distribution(distribution_update.distribution)
        
        # Update persona with new distribution
        persona.config_json = {
            **persona.config_json,
            "custom_distribution": distribution_update.distribution
        }
        
        # If batchId is provided, regenerate the batch with new distribution
        if distribution_update.batchId:
            batch = db.query(TransactionBatch).filter(
                TransactionBatch.id == distribution_update.batchId,
                TransactionBatch.user_id == user.id
            ).first()
            
            if not batch:
                raise HTTPException(status_code=404, detail="Batch not found")
            
            # Delete existing transactions
            db.query(Transaction).filter(Transaction.batch_id == batch.id).delete()
            db.query(TransactionExplanation).filter(TransactionExplanation.batch_id == batch.id).delete()
            db.query(BatchExplanation).filter(BatchExplanation.batch_id == batch.id).delete()
            
            # Regenerate transactions with new distribution
            dataset_path = persona.config_json.get("dataset_path")
            if not dataset_path:
                raise HTTPException(status_code=500, detail="No dataset path configured")
            
            # Get or create model
            model_key = f"persona_{persona_id}_{hash(dataset_path)}"
            model = models.get(model_key)
            
            if not model:
                X, C = data_processor.load_data(dataset_path, persona.name)
                input_dim = X.shape[1]
                condition_dim = C.shape[1]
                tensor_X, tensor_C = torch.FloatTensor(X), torch.FloatTensor(C)
                
                model = WGAN_GP(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    condition_dim=condition_dim,
                    device=data_processor.device
                )
                
                # Train model
                n_epochs = 15
                batch_size = 256
                for epoch in range(n_epochs):
                    total_loss = 0
                    for i in range(0, len(X), batch_size):
                        batch_X = tensor_X[i:i + batch_size]
                        batch_C = tensor_C[i:i + batch_size]
                        stats = model.train_step(batch_X, batch_C)
                        total_loss += stats['g_loss']
                    
                    if epoch % 5 == 0:
                        avg_loss = total_loss / (len(X) // batch_size)
                        logger.info(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")
                
                models[model_key] = model
            
            # Generate new transactions with updated distribution
            n_samples = 50 * batch.months
            categories = data_processor.category_columns
            
            # Use the new distribution
            category_weights = distribution_update.distribution
            total_weight = sum(category_weights.values())
            category_probs = {cat: w / total_weight for cat, w in category_weights.items()}
            # Deterministic assignment for exact proportions
            cats = list(category_probs.keys())
            probs = list(category_probs.values())
            counts = [int(round(p * n_samples)) for p in probs]
            # Adjust for rounding errors
            while sum(counts) < n_samples:
                counts[counts.index(max(counts))] += 1
            while sum(counts) > n_samples:
                counts[counts.index(max(counts))] -= 1
            chosen_categories = []
            for cat, count in zip(cats, counts):
                chosen_categories.extend([cat] * count)
            np.random.shuffle(chosen_categories)
            
            # Generate in batches
            batch_size = 100
            n_batches = (n_samples + batch_size - 1) // batch_size
            all_generated_data = []
            
            category_indices = {}
            for cat in set(chosen_categories):
                if cat.startswith('category_'):
                    category_name = cat
                else:
                    category_name = f"category_{cat}"
                
                try:
                    category_indices[cat] = list(categories).index(category_name)
                except ValueError:
                    try:
                        category_indices[cat] = list(categories).index(cat)
                    except ValueError:
                        logger.error(f"Could not find category index for {cat}")
                        raise ValueError(f"Invalid category: {cat}")
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_categories = chosen_categories[start_idx:end_idx]
                
                condition_matrix = np.zeros((len(batch_categories), len(categories)))
                for j, cat in enumerate(batch_categories):
                    idx = category_indices.get(cat)
                    if idx is not None:
                        condition_matrix[j, idx] = 1
                    else:
                        condition_matrix[j, random.randrange(len(categories))] = 1
                
                batch_generated = model.generate(len(batch_categories), condition_matrix)
                batch_combined = np.hstack([batch_generated, condition_matrix])
                all_generated_data.append(batch_combined)
            
            combined_data = np.vstack(all_generated_data)
            transactions = data_processor.inverse_transform(combined_data)
            
            # Insert new transactions
            db_transactions = []
            for tx in transactions:
                try:
                    booking_date = datetime.strptime(tx["bookingDateTime"], "%d/%m/%Y %H:%M:%S")
                except ValueError:
                    try:
                        booking_date = datetime.fromisoformat(tx["bookingDateTime"])
                    except ValueError:
                        logger.error(f"Invalid date format: {tx['bookingDateTime']}")
                        raise HTTPException(status_code=500, detail=f"Invalid date format: {tx['bookingDateTime']}")
                
                value_date = booking_date
                
                db_transactions.append(
                    Transaction(
                        batch_id=batch.id,
                        transaction_id=tx["transactionId"],
                        booking_date_time=booking_date,
                        value_date_time=value_date,
                        amount=float(tx["transactionAmount"]["amount"]),
                        currency=tx["transactionAmount"]["currency"],
                        creditor_name=tx["creditorName"],
                        creditor_account_iban=tx["creditorAccount"]["iban"],
                        debtor_name=tx["debtorName"],
                        debtor_account_iban=tx["debtorAccount"]["iban"],
                        remittance_information_unstructured=tx["remittanceInformationUnstructured"],
                        category=tx["category"]
                    )
                )
            
            db.bulk_save_objects(db_transactions)
            
            # Generate explanations
            explanation_service = ExplanationService(db)
            feature_importances = []
            for tx in transactions:
                tx_tensor, condition_tensor = data_processor.transaction_to_tensor(tx)
                with torch.no_grad():
                    _ = model.generator(tx_tensor, condition_tensor)
                    importance = model.generator.get_feature_importance()
                    importance_dict = {
                        'amount': float(importance[0][0]),
                        'day_of_month': float(importance[0][1])
                    }
                    for i, cat in enumerate(data_processor.category_columns):
                        category_name = cat.replace('category_', '')
                        importance_dict[category_name] = float(importance[0][i + 2])
                    feature_importances.append(importance_dict)
            
            explanation_service.process_batch(batch, feature_importances)
            
            # Update batch preview
            batch.preview_json = {"count": len(transactions)}
        
        db.commit()
        
        # Create audit log
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.DISTRIBUTION_UPDATED,
            entity_type="persona",
            entity_id=str(persona_id),
            details={
                "distribution_updated": True,
                "batch_regenerated": bool(distribution_update.batchId),
                "use_for_training": distribution_update.useForTraining
            }
        )
        db.add(audit_log)
        db.commit()
        
        return {
            "message": "Distribution updated successfully",
            "persona_id": persona_id,
            "batch_regenerated": bool(distribution_update.batchId)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating persona distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating persona distribution: {str(e)}")

@app.get("/transactions/{transaction_id}/explanation", tags=["explanations"])
async def get_transaction_explanation(
    transaction_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get AI-generated explanation for a specific transaction.
    
    Returns:
    - Feature importance scores
    - Applied transaction patterns
    - Natural language explanation
    - Confidence score
    - Additional metadata
    """
    # Find the transaction and verify ownership through batch
    transaction = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == transaction.batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    if not batch:
        raise HTTPException(status_code=403, detail="Not authorized to access this transaction")
    
    explanation = db.query(TransactionExplanation).filter(
        TransactionExplanation.transaction_id == transaction_id
    ).first()
    
    if not explanation:
        raise HTTPException(status_code=404, detail="No explanation found for this transaction")
    
    return {
        "transaction_id": transaction_id,
        "feature_importance": explanation.feature_importance,
        "applied_patterns": explanation.applied_patterns,
        "explanation_text": explanation.explanation_text,
        "confidence_score": explanation.confidence_score,
        "meta_info": explanation.meta_info
    }

@app.get("/batches/{batch_id}/evaluation", tags=["explanations"])
async def get_batch_evaluation(
    batch_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get evaluation metrics for a specific batch.
    
    Returns Inception Score, FID, and other quality metrics for the batch.
    """
    # Find the batch and verify ownership
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Get transactions for the batch
    transactions = db.query(Transaction).filter(
        Transaction.batch_id == batch_id
    ).all()
    
    if not transactions:
        raise HTTPException(status_code=404, detail="No transactions found for batch")
    
    # Convert to API format
    transaction_data = []
    for tx in transactions:
        transaction_data.append({
            "transactionId": tx.transaction_id,
            "bookingDateTime": tx.booking_date_time.isoformat(),
            "valueDateTime": tx.value_date_time.isoformat(),
            "transactionAmount": {
                "amount": str(tx.amount),
                "currency": tx.currency
            },
            "creditorName": tx.creditor_name,
            "creditorAccount": {"iban": tx.creditor_account_iban},
            "debtorName": tx.debtor_name,
            "debtorAccount": {"iban": tx.debtor_account_iban},
            "remittanceInformationUnstructured": tx.remittance_information_unstructured,
            "category": tx.category
        })
    
    try:
        # Initialize evaluator if needed
        global evaluator
        if evaluator is None:
            evaluator = TransactionEvaluator(device=data_processor.device)
        
        # Load real training data for comparison
        persona = db.query(Persona).filter(Persona.id == batch.persona_id).first()
        real_transactions = []
        
        if persona and persona.config_json.get("dataset_path"):
            try:
                dataset_path = persona.config_json["dataset_path"]
                X, C = data_processor.load_data(dataset_path, persona.name)
                real_data = data_processor.inverse_transform(np.hstack([X, C]))
                real_transactions = real_data[:min(len(real_data), 1000)]
            except Exception as e:
                logger.warning(f"Could not load real data for evaluation: {e}")
        
        # Calculate metrics
        metrics = evaluator.evaluate_batch(real_transactions, transaction_data)
        
        return {
            "batch_id": batch_id,
            "persona_name": persona.name if persona else "Unknown",
            "transaction_count": len(transaction_data),
            "evaluation_metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error calculating evaluation metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating evaluation metrics: {str(e)}")

def transaction_to_tensor(self, tx):
    """Convert a transaction to a tensor format suitable for the model."""
    # Extract numerical features
    try:
        if isinstance(tx, dict):
            amount = float(tx['transactionAmount']['amount'])
        else:
            amount = float(tx.amount)
    except (KeyError, AttributeError):
        logger.error(f"Invalid transaction format: {tx}")
        raise ValueError("Invalid transaction format - missing amount field")

    # Extract temporal features
    try:
        if isinstance(tx, dict):
            booking_date = pd.to_datetime(tx['bookingDateTime'])
        else:
            booking_date = pd.to_datetime(tx.booking_date_time)
    except (KeyError, AttributeError):
        logger.error(f"Invalid transaction format: {tx}")
        raise ValueError("Invalid transaction format - missing booking date field")

    day_of_month = booking_date.day / 31.0
    day_of_week = booking_date.dayofweek / 6.0
    
    # Create condition vector for category
    try:
        if isinstance(tx, dict):
            category = tx['category']
        else:
            category = tx.category
    except (KeyError, AttributeError):
        logger.error(f"Invalid transaction format: {tx}")
        raise ValueError("Invalid transaction format - missing category field")

    category_vector = np.zeros(len(self.categorical_columns))
    try:
        if category.startswith('category_'):
            category_name = category
        else:
            category_name = f"category_{category}"
        category_idx = list(self.categorical_columns).index(category_name)
        category_vector[category_idx] = 1
    except ValueError:
        # If category not found, use a random category
        category_vector[random.randrange(len(self.categorical_columns))] = 1
    
    # Combine features
    features = np.array([amount, day_of_month, day_of_week])
    
    # Normalize numerical features
    if not hasattr(self, '_feature_scaler'):
        self._feature_scaler = MinMaxScaler()
        self._feature_scaler.fit(np.array([[0, 0, 0], [10000, 1, 1]]))
    features = self._feature_scaler.transform(features.reshape(1, -1))[0]
    
    # Create tensors on CPU first
    input_tensor = torch.FloatTensor(features).unsqueeze(0)
    condition_tensor = torch.FloatTensor(category_vector).unsqueeze(0)
    
    # Move tensors to the correct device
    input_tensor = input_tensor.to(self.device)
    condition_tensor = condition_tensor.to(self.device)
    
    # Synchronize if using MPS device
    if self.device.type == 'mps':
        torch.mps.synchronize()
    
    return input_tensor, condition_tensor
