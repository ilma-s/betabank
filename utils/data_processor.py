import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from scipy.stats import entropy

class TransactionDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.numerical_columns = ['amount']
        self.categorical_columns = ['category']
        self.temporal_columns = ['bookingDateTime']
        
        # initialize dictionaries to store learned patterns
        self.merchant_by_category = defaultdict(list)
        self.descriptions_by_category = defaultdict(list)
        self.amount_ranges_by_category = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
        self.previous_batch = None
        self.VARIETY_THRESHOLD = 0.5

    def learn_patterns(self, data):
        """Learn patterns from training data"""
        for transaction in data:
            category = transaction['category']
            merchant = transaction['creditorName']
            description = transaction['remittanceInformationUnstructured']
            amount = float(transaction['transactionAmount']['amount'])

            # learn merchant patterns if not empty
            if merchant and merchant not in self.merchant_by_category[category]:
                self.merchant_by_category[category].append(merchant)

            # learn description patterns
            if description not in self.descriptions_by_category[category]:
                # store the description template by replacing the merchant name with {}
                if merchant:
                    description = description.replace(merchant, '{}')
                self.descriptions_by_category[category].append(description)

            # learn amount ranges
            self.amount_ranges_by_category[category]['min'] = min(
                self.amount_ranges_by_category[category]['min'], 
                amount
            )
            self.amount_ranges_by_category[category]['max'] = max(
                self.amount_ranges_by_category[category]['max'], 
                amount
            )

    def load_data(self, filepath):
        """Load and process training data"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # learn patterns from the training data
        self.learn_patterns(data)
        
        df = pd.DataFrame(data)
        
        # extract amount from nested structure
        df['amount'] = df['transactionAmount'].apply(lambda x: float(x['amount']))
        
        # convert datetime
        df['bookingDateTime'] = pd.to_datetime(df['bookingDateTime'])
        df['timestamp'] = df['bookingDateTime'].astype(np.int64) // 10**9
        
        # one-hot encode categories
        category_dummies = pd.get_dummies(df['category'], prefix='category')
        
        # normalize numerical features
        normalized_data = self.scaler.fit_transform(df[['amount', 'timestamp']])
        
        # combine features
        processed_data = np.hstack([
            normalized_data,
            category_dummies.values
        ])
        
        self.output_dim = processed_data.shape[1]
        self.category_columns = category_dummies.columns
        return processed_data

    def generate_transaction_details(self, category, amount):
        """Generate transaction details based on learned patterns"""
        import random
        
        # get merchants for this category
        merchants = self.merchant_by_category[category]
        descriptions = self.descriptions_by_category[category]
        
        if not merchants or not any(merchants):  
            merchant = ""
            description = random.choice(descriptions) if descriptions else f"Transaction - {category}"
        else:
            merchant = random.choice(merchants)
            # find a description template that can accommodate the merchant
            valid_descriptions = [d for d in descriptions if '{}' in d]
            if valid_descriptions:
                description = random.choice(valid_descriptions).format(merchant)
            else:
                description = f"Transaction at {merchant}"
            
        return merchant, description

    def inverse_transform(self, generated_data):
        """Transform generated data back into transaction format"""
        numerical_data = generated_data[:, :2]
        categorical_data = generated_data[:, 2:]
        
        denormalized_data = self.scaler.inverse_transform(numerical_data)
        
        transactions = []
        for i in range(len(generated_data)):
            category_idx = np.argmax(categorical_data[i])
            category = self.category_columns[category_idx].replace('category_', '')
            amount = abs(denormalized_data[i, 0])
            
            merchant_name, description = self.generate_transaction_details(category, amount)
            
            timestamp = int(denormalized_data[i, 1])
            booking_date = datetime.fromtimestamp(timestamp)
            
            transaction = {
                "transactionId": f"TX{np.random.randint(100000, 999999)}",
                "bookingDateTime": booking_date.isoformat(),
                "valueDateTime": booking_date.isoformat(),
                "transactionAmount": {
                    "amount": f"{amount:.2f}",
                    "currency": "EUR"
                },
                "creditorName": merchant_name,
                "creditorAccount": {
                    "iban": f"DE{np.random.randint(1000000000, 9999999999)}"
                },
                "debtorName": f"Person{np.random.randint(1000, 9999)}",
                "debtorAccount": {
                    "iban": f"DE{np.random.randint(1000000000, 9999999999)}"
                },
                "remittanceInformationUnstructured": description,
                "category": category
            }
            transactions.append(transaction)
            
        return transactions 

    def calculate_batch_features(self, transactions):
        """Calculate statistical features for a batch of transactions"""
        amounts = [float(tx['transactionAmount']['amount']) for tx in transactions]
        dates = [datetime.fromisoformat(tx['bookingDateTime']) for tx in transactions]
        categories = [tx['category'] for tx in transactions]
        
        # amount statistics
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        
        # time spread in days
        time_spread = (max(dates) - min(dates)).days
        
        # category entropy
        category_counts = defaultdict(int)
        for cat in categories:
            category_counts[cat] += 1
        probs = [count/len(categories) for count in category_counts.values()]
        category_entropy = entropy(probs)

        # transaction patterns
        merchant_diversity = len(set(tx['creditorName'] for tx in transactions if tx['creditorName']))
        
        return {
            'mean_amount': mean_amount,
            'std_amount': std_amount,
            'time_spread': time_spread,
            'category_entropy': category_entropy,
            'merchant_diversity': merchant_diversity
        }

    def calculate_frechet_distance(self, features1, features2):
        """Calculate weighted Fréchet-inspired distance between feature sets"""
        weights = {
            'mean_amount': 1.0,
            'std_amount': 0.8,
            'time_spread': 0.6,
            'category_entropy': 1.0,
            'merchant_diversity': 0.7
        }
        
        distance = 0
        for key in weights:
            distance += weights[key] * (features1[key] - features2[key])**2
        
        return np.sqrt(distance)

    def check_transaction_variety(self, new_transactions):
        """Check if new transactions are sufficiently different from the previous batch"""
        if self.previous_batch is None:
            self.previous_batch = new_transactions
            return True
        
        new_features = self.calculate_batch_features(new_transactions)
        prev_features = self.calculate_batch_features(self.previous_batch)
        
        distance = self.calculate_frechet_distance(new_features, prev_features)
        
        if distance < self.VARIETY_THRESHOLD:
            return False
        
        self.previous_batch = new_transactions
        return True 