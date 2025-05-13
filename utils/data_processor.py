import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from scipy.stats import entropy
from collections import Counter
import random

class TransactionDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler() # initialize scaler for normalization; scale numerical values between 0 and 1
        self.numerical_columns = ['amount']
        self.categorical_columns = ['category']
        self.temporal_columns = ['bookingDateTime']
        
        # initialize dictionaries to store learned patterns
        self.merchant_by_category = defaultdict(list)
        self.descriptions_by_category = defaultdict(list)
        self.amount_ranges_by_category = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
        self.fixed_price_merchants = defaultdict(set)
        self.merchant_frequency = defaultdict(lambda: defaultdict(list))  # track merchant transaction dates
        self.previous_batch = None
        self.VARIETY_THRESHOLD = 0.5

    def validate_temporal_patterns(self, data):
        """Validate temporal patterns in the data (salary and subscriptions/utilities payments)"""
        # reset merchant frequency tracking
        self.merchant_frequency.clear()
        
        # track transactions by merchant and month
        for transaction in data:
            merchant = transaction['creditorName']
            category = transaction['category']
            date = pd.to_datetime(transaction['bookingDateTime'])
            month_key = f"{date.year}-{date.month}"
            
            # add transaction to merchant's monthly records
            self.merchant_frequency[merchant][month_key].append({
                'date': date,
                'category': category,
                'amount': float(transaction['transactionAmount']['amount'])
            })
        
        issues = []
        
        # Check patterns
        for merchant, monthly_transactions in self.merchant_frequency.items():
            for month, transactions in monthly_transactions.items():
                # group by category
                category_counts = defaultdict(int)
                for trans in transactions:
                    category_counts[trans['category']] += 1
                
                # validate patterns
                for category, count in category_counts.items():
                    if category == "Salary" and count > 1:
                        issues.append(f"Multiple salary payments from {merchant} in {month}")
                    
                    if category in ["Subscriptions", "Utilities"] and count > 1:
                        issues.append(f"Multiple {category.lower()} payments to {merchant} in {month}")
        
        return issues

    def learn_patterns(self, data):
        """Learn patterns from training data"""
        merchant_amounts = defaultdict(set)
        
        # first validate temporal patterns
        issues = self.validate_temporal_patterns(data)
        if issues:
            print("Temporal pattern issues found:")
            for issue in issues:
                print(f"- {issue}")
        
        for transaction in data:
            category = transaction['category']
            merchant = transaction['creditorName']
            description = transaction['remittanceInformationUnstructured']
            amount = float(transaction['transactionAmount']['amount'])
            
            # learn merchant patterns if not empty
            if merchant:
                if merchant not in self.merchant_by_category[category]:
                    self.merchant_by_category[category].append(merchant)
                merchant_amounts[merchant].add(amount)
            
            # learn description patterns
            if description not in self.descriptions_by_category[category]:
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
        
        # update fixed price merchants
        for merchant, amounts in merchant_amounts.items():
            if len(amounts) == 1:  # if merchant only has one price
                self.fixed_price_merchants[merchant] = amounts.pop()

    def balance_categories(self, data, target_count=None):
        """Oversample rare categories so each has a similar number of samples."""
        categories = [tx['category'] for tx in data]
        counts = Counter(categories)
        max_count = target_count or max(counts.values())
        balanced = []
        for cat in counts:
            cat_txs = [tx for tx in data if tx['category'] == cat]
            if len(cat_txs) < max_count:
                cat_txs = cat_txs * (max_count // len(cat_txs)) + random.choices(cat_txs, k=max_count % len(cat_txs))
            balanced.extend(cat_txs)
        random.shuffle(balanced)
        return balanced
    
    def load_data(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        data = self.balance_categories(data)
        self.learn_patterns(data)
        
        df = pd.DataFrame(data)
        df['amount'] = df['transactionAmount'].apply(lambda x: float(x['amount']))
        df['bookingDateTime'] = pd.to_datetime(df['bookingDateTime'])
        df['timestamp'] = df['bookingDateTime'].astype(np.int64) // 10**9

        # Encode and normalize
        category_dummies = pd.get_dummies(df['category'], prefix='category')
        self.category_columns = category_dummies.columns
        self.condition_dim = category_dummies.shape[1]

        normalized = self.scaler.fit_transform(df[['amount', 'timestamp']])
        X = normalized
        C = category_dummies.values

        self.output_dim = X.shape[1]  # For critic input
        return X, C

    def generate_transaction_details(self, category, amount):
        """Generate transaction details based on learned patterns"""
        import random
        
        merchants = self.merchant_by_category[category]
        descriptions = self.descriptions_by_category[category]
        
        if not merchants or not any(merchants):  
            merchant = ""
            description = random.choice(descriptions) if descriptions else f"Transaction - {category}"
        else:
            merchant = random.choice(merchants)
            # use fixed price if merchant has one
            if merchant in self.fixed_price_merchants:
                amount = self.fixed_price_merchants[merchant]
            
            # find a description template that can accommodate the merchant
            valid_descriptions = [d for d in descriptions if '{}' in d]
            if valid_descriptions:
                description = random.choice(valid_descriptions).format(merchant)
            else:
                description = f"Transaction at {merchant}"
            
        return merchant, description, amount

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
            
            merchant_name, description, amount = self.generate_transaction_details(category, amount)
            
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

    def check_transaction_variety(self, transactions):
        """Enhanced check for transaction variety including category distribution."""
        if not transactions:
            return False
        
        # Basic variety check: ensure at least 2 categories and at least 2 unique merchants
        categories = set(tx['category'] for tx in transactions)
        merchants = set(tx['creditorName'] for tx in transactions if tx['creditorName'])
        if len(categories) < 2 or len(merchants) < 2:
            return False
        
        # Check category distribution
        category_dist = self.get_category_distribution(transactions)
        if len(category_dist) < 5:  # Should have at least 5 different categories
            return False
        
        # Check for reasonable distribution of amounts
        amounts = [float(tx['transactionAmount']['amount']) for tx in transactions]
        if max(amounts) / min(amounts) < 10:  # Should have at least 10x difference between min and max
            return False
        
        return True

    def get_category_distribution(self, transactions):
        """Calculate the distribution of transaction categories."""
        category_counts = {}
        for tx in transactions:
            category = tx.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        total = len(transactions)
        return {
            category: {
                'count': count,
                'percentage': (count / total) * 100
            }
            for category, count in category_counts.items()
        }

    def ensure_category_distribution(self, persona_type):
        """Get target distribution based on persona type."""
        target_distributions = {
            'gambling addict': {
                'Gambling': {'min': 30, 'max': 40},
                'Shopping': {'min': 10, 'max': 15},
                'Dining': {'min': 10, 'max': 15},
                'Transport': {'min': 5, 'max': 10},
                'Groceries': {'min': 5, 'max': 10},
                'Utilities': {'min': 5, 'max': 8},
                'Subscriptions': {'min': 3, 'max': 5},
                'ATM Withdrawals': {'min': 10, 'max': 15},
                'Salary': {'min': 2, 'max': 4},
                'Refunds': {'min': 1, 'max': 3}
            },
            'shopping addict': {
                'Shopping': {'min': 35, 'max': 45},
                'Dining': {'min': 10, 'max': 15},
                'Transport': {'min': 5, 'max': 10},
                'Groceries': {'min': 10, 'max': 15},
                'Utilities': {'min': 5, 'max': 8},
                'Subscriptions': {'min': 3, 'max': 5},
                'ATM Withdrawals': {'min': 5, 'max': 10},
                'Salary': {'min': 2, 'max': 4},
                'Refunds': {'min': 3, 'max': 5}
            },
            'crypto enthusiast': {
                'Crypto': {'min': 35, 'max': 45},
                'Shopping': {'min': 10, 'max': 15},
                'Dining': {'min': 5, 'max': 10},
                'Transport': {'min': 5, 'max': 10},
                'Groceries': {'min': 5, 'max': 10},
                'Utilities': {'min': 5, 'max': 8},
                'Subscriptions': {'min': 3, 'max': 5},
                'ATM Withdrawals': {'min': 5, 'max': 10},
                'Salary': {'min': 2, 'max': 4},
                'Refunds': {'min': 1, 'max': 3}
            },
            'money mule': {
                'ATM Withdrawals': {'min': 30, 'max': 40},
                'Shopping': {'min': 15, 'max': 20},
                'Dining': {'min': 5, 'max': 10},
                'Transport': {'min': 5, 'max': 10},
                'Groceries': {'min': 5, 'max': 10},
                'Utilities': {'min': 5, 'max': 8},
                'Subscriptions': {'min': 3, 'max': 5},
                'Salary': {'min': 2, 'max': 4},
                'Refunds': {'min': 1, 'max': 3}
            }
        }

        return target_distributions.get(persona_type.lower(), {}) 