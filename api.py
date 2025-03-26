from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import json
import logging
import torch

from models.wgan_gp import WGAN_GP
from utils.data_processor import TransactionDataProcessor
from utils.database import get_db
from models.database_models import User, Persona, TransactionBatch, Transaction

from passlib.context import CryptContext
from jose import JWTError, jwt

# JWT Configuration
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predefined personas and their datasets
PERSONAS = {
    "crypto_enthusiast": {
        "name": "Crypto Enthusiast",
        "description": "A tech-savvy individual who primarily invests in cryptocurrencies",
        "dataset_path": "testing_datasets/crypto_enthusiast.json"
    },
    "shopping_addict": {
        "name": "Shopping Addict",
        "description": "Someone who frequently shops online and at retail stores",
        "dataset_path": "testing_datasets/shopping_addict.json"
    },
    "gambling_addict": {
        "name": "Gambling Addict",
        "description": "A person with a regular gambling habit",
        "dataset_path": "testing_datasets/gambling_addict.json"
    },
    "money_mule": {
        "name": "Money Mule",
        "description": "Individual involved in moving illegally acquired money",
        "dataset_path": "testing_datasets/money_mule.json"
    }
}

# Store trained models in memory
models = {}
data_processor = TransactionDataProcessor()

# User authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
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

# API Endpoints
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(username: str, password: str, db: Session = Depends(get_db)):
    # Check if username already exists
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Create new user
    hashed_password = get_password_hash(password)
    new_user = User(username=username, password_hash=hashed_password)
    db.add(new_user)
    
    # Add default personas for the user
    for persona_id, persona_data in PERSONAS.items():
        new_persona = Persona(
            user_id=new_user.id,
            name=persona_data["name"],
            description=persona_data["description"],
            config_json={"dataset_path": persona_data["dataset_path"]}
        )
        db.add(new_persona)
    
    db.commit()
    return {"message": "User registered successfully"}

@app.get("/personas")
async def get_personas(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    personas = db.query(Persona).filter(Persona.user_id == user.id).all()
    return {"personas": [{"id": str(p.id), "name": p.name, "description": p.description} for p in personas]}

@app.post("/personas")
async def create_persona(name: str, description: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_persona = Persona(
        user_id=user.id,
        name=name,
        description=description,
        config_json={}
    )
    db.add(new_persona)
    db.commit()
    db.refresh(new_persona)
    return {"id": new_persona.id, "name": new_persona.name, "description": new_persona.description}

@app.get("/generate/{persona_id}")
async def generate_transactions(
    persona_id: int, 
    batch_name: Optional[str] = None,
    n_samples: int = 20,
    user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    # Check if persona exists and belongs to user
    persona = db.query(Persona).filter(Persona.id == persona_id, Persona.user_id == user.id).first()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    # Get dataset path from persona config or use default
    dataset_path = None
    if persona.config_json and "dataset_path" in persona.config_json:
        dataset_path = persona.config_json["dataset_path"]
    else:
        # Use a default dataset if none is configured
        for p_id, p_data in PERSONAS.items():
            if p_data["name"].lower() == persona.name.lower():
                dataset_path = p_data["dataset_path"]
                break
        
        if not dataset_path:
            dataset_path = PERSONAS["shopping_addict"]["dataset_path"]  # Default fallback
    
    max_attempts = 5
    attempts = 0
    
    while attempts < max_attempts:
        processed_data = data_processor.load_data(dataset_path)
        model_key = f"persona_{persona_id}"
        model = models.get(model_key)
        
        if not model:
            logger.info(f"Training model for persona {persona_id}...")
            model = WGAN_GP(input_dim=100, output_dim=data_processor.output_dim)
            tensor_data = torch.FloatTensor(processed_data)
            for epoch in range(100):
                stats = model.train_step(tensor_data)
                if epoch % 10 == 0:
                    logger.info(f"Training persona {persona_id} - Epoch {epoch}: {stats}")
            models[model_key] = model
        
        generated_data = model.generate(n_samples)
        transactions = data_processor.inverse_transform(generated_data)
        
        # Check variety using Fréchet-inspired distance
        if data_processor.check_transaction_variety(transactions):
            # Create transaction batch
            if not batch_name:
                batch_name = f"Batch {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create a preview (first 3 transactions)
            preview = transactions[:3] if len(transactions) >= 3 else transactions
            
            # Create transaction batch record
            new_batch = TransactionBatch(
                user_id=user.id,
                persona_id=persona_id,
                name=batch_name,
                preview_json=preview,
                summary_json={
                    "count": len(transactions),
                    "generated_at": datetime.now().isoformat()
                }
            )
            db.add(new_batch)
            db.flush()  # Get the batch ID without committing transaction
            
            # Store individual transactions
            for tx in transactions:
                new_tx = Transaction(
                    batch_id=new_batch.id,
                    transaction_id=tx["transactionId"],
                    booking_date_time=datetime.fromisoformat(tx["bookingDateTime"]),
                    value_date_time=datetime.fromisoformat(tx["valueDateTime"]),
                    amount=float(tx["transactionAmount"]["amount"]),
                    currency=tx["transactionAmount"]["currency"],
                    creditor_name=tx["creditorName"],
                    creditor_account_iban=tx["creditorAccount"]["iban"],
                    debtor_name=tx["debtorName"],
                    debtor_account_iban=tx["debtorAccount"]["iban"],
                    remittance_information_unstructured=tx["remittanceInformationUnstructured"],
                    category=tx["category"]
                )
                db.add(new_tx)
            
            db.commit()
            return {"batch_id": new_batch.id, "transactions": transactions}
        
        attempts += 1
        logger.info(f"Generated transactions too similar, attempt {attempts}/{max_attempts}")
    
    # If sufficiently different transactions couldn't be generated after max attempts
    logger.warning("Could not generate sufficiently different transactions")
    return {"transactions": transactions}  # Return last generated batch anyway

@app.get("/batches")
async def get_transaction_batches(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    batches = db.query(TransactionBatch).filter(TransactionBatch.user_id == user.id).order_by(TransactionBatch.created_at.desc()).all()
    
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
            "preview": batch.preview_json
        })
    
    return {"batches": result}

@app.get("/batches/{batch_id}")
async def get_batch_transactions(batch_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if batch exists and belongs to user
    batch = db.query(TransactionBatch).filter(TransactionBatch.id == batch_id, TransactionBatch.user_id == user.id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Retrieve transactions
    transactions = db.query(Transaction).filter(Transaction.batch_id == batch_id).all()
    
    # Format transactions for response
    formatted_transactions = []
    for tx in transactions:
        formatted_transactions.append({
            "transactionId": tx.transaction_id,
            "bookingDateTime": tx.booking_date_time.isoformat(),
            "valueDateTime": tx.value_date_time.isoformat(),
            "transactionAmount": {
                "amount": str(tx.amount),
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
            "category": tx.category,
            "edited": tx.edited
        })
    
    # Get persona info
    persona = db.query(Persona).filter(Persona.id == batch.persona_id).first()
    
    return {
        "batch": {
            "id": batch.id,
            "name": batch.name,
            "persona_id": batch.persona_id,
            "persona_name": persona.name if persona else "Unknown",
            "created_at": batch.created_at.isoformat(),
        },
        "transactions": formatted_transactions
    } 