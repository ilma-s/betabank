from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.wgan_gp import WGAN_GP
from utils.data_processor import TransactionDataProcessor
import torch
import logging

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

PERSONAS = {
    "crypto_enthusiast": "testing_datasets/crypto_enthusiast.json",
    "shopping_addict": "testing_datasets/shopping_addict.json",
    "gambling_addict": "testing_datasets/gambling_addict.json",
    "money_mule": "testing_datasets/money_mule.json"
}

# store trained models in memory
models = {}
data_processor = TransactionDataProcessor()

@app.get("/personas")
async def get_personas():
    return {"personas": [{"id": k, "name": k.replace("_", " ").title()} for k in PERSONAS.keys()]}

@app.get("/generate/{persona_type}")
async def generate_transactions(persona_type: str, n_samples: int = 10):
    if persona_type not in PERSONAS:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    max_attempts = 5
    attempts = 0
    
    while attempts < max_attempts:
        processed_data = data_processor.load_data(PERSONAS[persona_type])
        model = models.get(persona_type)
        
        if not model:
            logger.info(f"Training model for {persona_type}...")
            model = WGAN_GP(input_dim=100, output_dim=data_processor.output_dim)
            tensor_data = torch.FloatTensor(processed_data)
            for epoch in range(100):
                stats = model.train_step(tensor_data)
                if epoch % 10 == 0:
                    logger.info(f"Training {persona_type} - Epoch {epoch}: {stats}")
            models[persona_type] = model
        
        generated_data = model.generate(n_samples)
        transactions = data_processor.inverse_transform(generated_data)
        
        # check variety using Fréchet-inspired distance
        if data_processor.check_transaction_variety(transactions):
            return {"transactions": transactions}
            
        attempts += 1
        logger.info(f"Generated transactions too similar, attempt {attempts}/{max_attempts}")
    
    # if sufficiently different transactions couldn't be generated after max attempts
    logger.warning("Could not generate sufficiently different transactions")
    return {"transactions": transactions}  # return last generated batch anyway 