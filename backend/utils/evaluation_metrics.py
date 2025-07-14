import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict
import logging
import datetime

logger = logging.getLogger(__name__)

class TransactionInceptionNet(nn.Module):
    """
    An enhanced Inception-like network for transaction data evaluation.
    This network learns to classify transaction patterns and assess quality with improved architecture.
    """
    def __init__(self, input_dim=10, num_classes=15):
        super(TransactionInceptionNet, self).__init__()
        
        # Enhanced feature extraction layers with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Enhanced classification head with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
        
        # Enhanced feature extraction for FID with deeper network
        self.feature_extractor_fid = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Initialize weights for better training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        class_probs = self.classifier(features)
        fid_features = self.feature_extractor_fid(features)
        return class_probs, fid_features

class TransactionEvaluator:
    """
    Evaluator for transaction data quality using Inception Score and FID.
    """
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device('cpu')
        self.inception_net = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)

    def parse_date(self, date_str):
        """
        Parse a date string in ISO or DD/MM/YYYY HH:MM:SS format.
        Returns a numpy.datetime64 or raises ValueError if parsing fails.
        """
        try:
            # Try ISO format first
            return np.datetime64(date_str)
        except Exception:
            pass
        try:
            # Try DD/MM/YYYY HH:MM:SS
            dt = datetime.datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
            return np.datetime64(dt)
        except Exception:
            pass
        # Fallback: try parsing just the date
        try:
            dt = datetime.datetime.strptime(date_str, "%d/%m/%Y")
            return np.datetime64(dt)
        except Exception:
            pass
        raise ValueError(f"Unrecognized date format: {date_str}")

    def prepare_transaction_features(self, transactions):
        """
        Convert transactions to feature vectors for evaluation.
        """
        features = []
        
        for tx in transactions:
            # Extract numerical features
            amount = float(tx['transactionAmount']['amount'])
            
            # Parse date
            try:
                date = self.parse_date(tx['bookingDateTime'])
                day_of_month = date.astype('datetime64[D]').astype(int) % 31
                day_of_week = date.astype('datetime64[D]').astype(int) % 7
                hour = date.astype('datetime64[h]').astype(int) % 24
            except Exception:
                day_of_month = 15  # Default values
                day_of_week = 3
                hour = 12
            
            # Category encoding (one-hot like)
            category = tx.get('category', 'Unknown')
            category_hash = hash(category) % 10  # Simple hash for category
            
            # Merchant diversity
            merchant = tx.get('creditorName', 'Unknown')
            merchant_hash = hash(merchant) % 10
            
            # Create feature vector
            feature_vector = [
                amount / 1000.0,  # Normalized amount
                day_of_month / 31.0,
                day_of_week / 7.0,
                hour / 24.0,
                category_hash / 10.0,
                merchant_hash / 10.0,
                abs(amount) / 1000.0,  # Absolute amount
                (amount > 0) * 1.0,  # Is income
                (amount < 0) * 1.0,  # Is expense
                len(str(amount)) / 10.0  # Amount precision
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_inception_net(self, real_transactions, epochs=100):  # Increased from 50
        """
        Train the inception network on real transaction data with improved training.
        """
        features = self.prepare_transaction_features(real_transactions)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Data augmentation for better diversity
        augmented_features = self._augment_features(features_scaled)
        
        # Create synthetic labels based on transaction patterns
        labels = self._create_synthetic_labels(augmented_features)
        
        # Initialize and train the network with better architecture
        self.inception_net = TransactionInceptionNet(input_dim=10, num_classes=15).to(self.device)  # Increased classes
        
        # Convert to tensors
        X = torch.FloatTensor(augmented_features).to(self.device)
        y = torch.LongTensor(labels).to(self.device)
        
        # Training setup with better parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.inception_net.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR, add weight decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        self.inception_net.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs, _ = self.inception_net(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.inception_net.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            if epoch % 10 == 0:
                logger.info(f"Inception training epoch {epoch}, loss: {loss.item():.4f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:  # Early stopping
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    def _augment_features(self, features):
        """
        Augment features to improve diversity and training quality.
        """
        augmented = []
        for feature in features:
            # Original feature
            augmented.append(feature)
            
            # Add noise for robustness
            noise = np.random.normal(0, 0.01, feature.shape)
            augmented.append(feature + noise)
            
            # Slight variations
            variation = feature * (1 + np.random.normal(0, 0.05))
            augmented.append(variation)
        
        return np.array(augmented)

    def _create_synthetic_labels(self, features):
        """
        Create synthetic labels based on transaction patterns with improved categorization.
        """
        labels = []
        
        for feature in features:
            amount, day_of_month, day_of_week, hour, category_hash, merchant_hash, abs_amount, is_income, is_expense, precision = feature
            
            # More sophisticated labeling for better Inception Score
            if is_income > 0.5:
                if abs_amount > 0.8:
                    label = 0  # High-value income
                else:
                    label = 1  # Regular income
            elif abs_amount > 0.7:
                if is_expense > 0.5:
                    label = 2  # Large expenses
                else:
                    label = 3  # Large transactions
            elif day_of_week < 0.3:
                label = 4  # Weekend transactions
            elif hour < 0.3 or hour > 0.7:
                label = 5  # Off-hours transactions
            elif category_hash < 0.2:
                label = 6  # Very common categories
            elif category_hash < 0.4:
                label = 7  # Common categories
            elif merchant_hash < 0.3:
                label = 8  # Common merchants
            elif precision > 0.6:
                label = 9  # High precision amounts
            elif day_of_month < 0.2:
                label = 10  # Early month transactions
            elif day_of_month > 0.8:
                label = 11  # Late month transactions
            elif abs_amount < 0.2:
                label = 12  # Small transactions
            elif is_expense > 0.5:
                label = 13  # Regular expenses
            else:
                label = 14  # Other transactions
            
            labels.append(label)
        
        return labels
    
    def calculate_inception_score(self, generated_transactions, splits=10):
        """
        Calculate Inception Score for generated transactions with improved methodology.
        Higher score indicates better quality and diversity.
        """
        if self.inception_net is None:
            logger.warning("Inception network not trained. Returning default score.")
            return 1.0
        
        self.inception_net.eval()
        
        features = self.prepare_transaction_features(generated_transactions)
        features_scaled = self.scaler.transform(features)
        
        with torch.no_grad():
            X = torch.FloatTensor(features_scaled).to(self.device)
            class_probs, _ = self.inception_net(X)
            class_probs = class_probs.cpu().numpy()
        
        # Ensure probabilities are valid
        class_probs = np.clip(class_probs, 1e-8, 1.0)
        
        # Calculate Inception Score with improved methodology
        scores = []
        split_size = max(1, len(class_probs) // splits)
        
        for i in range(splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < splits - 1 else len(class_probs)
            
            if start_idx >= len(class_probs):
                break
                
            split_probs = class_probs[start_idx:end_idx]
            
            # Calculate mean probability distribution
            mean_probs = np.mean(split_probs, axis=0)
            
            # Ensure mean_probs is valid
            mean_probs = np.clip(mean_probs, 1e-8, 1.0)
            mean_probs = mean_probs / np.sum(mean_probs)  # Renormalize
            
            # Calculate KL divergence with improved numerical stability
            kl_div = np.sum(split_probs * np.log(split_probs / mean_probs + 1e-8), axis=1)
            
            # Calculate exponential of mean KL divergence
            score = np.exp(np.mean(kl_div))
            scores.append(score)
        
        # Return mean score, but ensure we have valid scores
        if not scores:
            return 1.0
        
        final_score = np.mean(scores)
        
        # Additional quality checks
        if final_score < 1.0:
            logger.warning(f"Low Inception Score detected: {final_score:.4f}")
        elif final_score > 10.0:
            logger.warning(f"Unusually high Inception Score detected: {final_score:.4f}")
        
        return final_score
    
    def calculate_fid(self, real_transactions, generated_transactions):
        """
        Calculate Fréchet Inception Distance between real and generated transactions.
        Lower score indicates better quality.
        """
        if self.inception_net is None:
            logger.warning("Inception network not trained. Returning default FID.")
            return 100.0
        
        self.inception_net.eval()
        
        # Prepare features
        real_features = self.prepare_transaction_features(real_transactions)
        gen_features = self.prepare_transaction_features(generated_transactions)
        
        real_features_scaled = self.scaler.transform(real_features)
        gen_features_scaled = self.scaler.transform(gen_features)
        
        with torch.no_grad():
            real_X = torch.FloatTensor(real_features_scaled).to(self.device)
            gen_X = torch.FloatTensor(gen_features_scaled).to(self.device)
            
            _, real_fid_features = self.inception_net(real_X)
            _, gen_fid_features = self.inception_net(gen_X)
            
            real_fid_features = real_fid_features.cpu().numpy()
            gen_fid_features = gen_fid_features.cpu().numpy()
        
        # Calculate FID
        mu_real = np.mean(real_fid_features, axis=0)
        mu_gen = np.mean(gen_fid_features, axis=0)
        
        sigma_real = np.cov(real_fid_features, rowvar=False)
        sigma_gen = np.cov(gen_fid_features, rowvar=False)
        
        # FID calculation
        diff = mu_real - mu_gen
        covmean = sigma_real @ sigma_gen
        
        # Check if the product is positive definite
        if np.any(np.linalg.eigvals(covmean) <= 0):
            # Use a small regularization
            covmean += np.eye(covmean.shape[0]) * 1e-6
        
        fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
        
        return np.real(fid)
    
    def calculate_diversity_score(self, transactions):
        """
        Calculate diversity score based on transaction variety.
        """
        if not transactions:
            return 0.0
        
        categories = set(tx.get('category', 'Unknown') for tx in transactions)
        merchants = set(tx.get('creditorName', 'Unknown') for tx in transactions)
        amounts = [float(tx['transactionAmount']['amount']) for tx in transactions]
        
        # Category diversity
        category_diversity = len(categories) / 10.0  # Normalize by expected max categories
        
        # Merchant diversity
        merchant_diversity = len(merchants) / 20.0  # Normalize by expected max merchants
        
        # Amount diversity (coefficient of variation)
        amount_cv = np.std(amounts) / (np.mean(amounts) + 1e-8)
        amount_diversity = min(amount_cv / 2.0, 1.0)  # Normalize
        
        # Temporal diversity
        try:
            dates = [self.parse_date(tx['bookingDateTime']) for tx in transactions]
            unique_days = len(set(date.astype('datetime64[D]') for date in dates))
            temporal_diversity = min(unique_days / 30.0, 1.0)  # Normalize by month
        except Exception:
            temporal_diversity = 0.5
        
        # Combined diversity score
        diversity_score = (category_diversity + merchant_diversity + 
                          amount_diversity + temporal_diversity) / 4.0
        
        return diversity_score
    
    def calculate_realism_score(self, transactions):
        """
        Calculate realism score based on transaction patterns.
        """
        if not transactions:
            return 0.0
        
        scores = []
        
        for tx in transactions:
            amount = float(tx['transactionAmount']['amount'])
            category = tx.get('category', 'Unknown')
            
            # Check for realistic patterns
            score = 0.0
            
            # Amount realism
            if 0.01 <= abs(amount) <= 10000:
                score += 0.3
            elif 0.001 <= abs(amount) <= 100000:
                score += 0.2
            else:
                score += 0.1
            
            # Category realism
            realistic_categories = {
                'Shopping', 'Dining', 'Transport', 'Groceries', 'Utilities',
                'Subscriptions', 'Salary', 'Refunds', 'ATM Withdrawals',
                'Gambling', 'Crypto'
            }
            if category in realistic_categories:
                score += 0.3
            else:
                score += 0.1
            
            # Temporal realism
            try:
                date = self.parse_date(tx['bookingDateTime'])
                hour = date.astype('datetime64[h]').astype(int) % 24
                if 6 <= hour <= 23:  # Reasonable hours
                    score += 0.2
                else:
                    score += 0.1
            except Exception:
                score += 0.1
            
            # Merchant realism
            merchant = tx.get('creditorName', '')
            if len(merchant) > 3 and len(merchant) < 50:
                score += 0.2
            else:
                score += 0.1
            
            scores.append(score)
        
        return np.mean(scores)
    
    def evaluate_batch(self, real_transactions, generated_transactions):
        """
        Comprehensive evaluation of generated transactions.
        """
        # Train inception network if not already trained
        if self.inception_net is None and real_transactions:
            logger.info("Training Inception network for evaluation...")
            self.train_inception_net(real_transactions)
        
        # Calculate metrics
        inception_score = self.calculate_inception_score(generated_transactions)
        fid_score = self.calculate_fid(real_transactions, generated_transactions) if real_transactions else 100.0
        diversity_score = self.calculate_diversity_score(generated_transactions)
        realism_score = self.calculate_realism_score(generated_transactions)
        
        return {
            'inception_score': inception_score,
            'fid_score': fid_score,
            'diversity_score': diversity_score,
            'realism_score': realism_score,
            'overall_score': (inception_score + (100 - fid_score) / 100 + diversity_score + realism_score) / 4
        } 