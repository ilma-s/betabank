import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from scipy.stats import entropy
from collections import Counter
import random
import boto3
import os
from urllib.parse import urlparse
import torch

class TransactionDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler() # initialize scaler for normalization; scale numerical values between 0 and 1
        self.numerical_columns = ['amount']
        self.categorical_columns = ['category']
        self.temporal_columns = ['bookingDateTime']
        
        # Force CPU device for better compatibility
        self.device = torch.device("cpu")
        
        # Define income categories
        self.income_categories = {'Salary', 'Refunds', 'Investment Returns', 'Interest', 'Deposits'}
        
        # Define subscription merchants with their fixed monthly amounts
        self.subscription_merchants = {
            'Netflix': 14.99,
            'Spotify': 9.99,
            'Amazon Prime': 7.99,
            'Disney+': 8.99,
            'Apple Music': 9.99,
            'HBO Max': 14.99,
            'YouTube Premium': 11.99,
            'Microsoft 365': 6.99,
            'PlayStation Plus': 9.99,
            'Xbox Game Pass': 9.99
        }

        # Define utility merchants with their fixed monthly amounts
        self.utility_merchants = {
            'City Power': 89.99,  # Electricity
            'Water Corp': 45.99,  # Water
            'Gas Connect': 65.99,  # Gas
            'Internet Plus': 49.99,  # Internet
            'Mobile Network': 29.99,  # Mobile
            'Waste Management': 19.99,  # Waste collection
            'Home Insurance Co': 35.99,  # Home insurance
            'Security Systems': 25.99,  # Security system
        }
        
        # initialize dictionaries to store learned patterns
        self.merchant_by_category = defaultdict(list)
        self.descriptions_by_category = defaultdict(list)
        self.amount_ranges_by_category = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
        self.fixed_price_merchants = defaultdict(set)
        self.merchant_frequency = defaultdict(lambda: defaultdict(list))  # track merchant transaction dates
        self.previous_batch = None
        self.VARIETY_THRESHOLD = 0.5
        
        # Initialize S3 client if credentials are available
        self.s3_client = None
        required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        
        # Debug: Print environment variables status
        print("Checking AWS credentials:")
        for var in required_vars:
            print(f"{var} present: {var in os.environ}")
        
        if all(k in os.environ for k in required_vars):
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                    region_name=os.environ.get('AWS_REGION', 'eu-west-2')
                )
                print("Successfully initialized S3 client")
            except Exception as e:
                print(f"Error initializing S3 client: {str(e)}")
        else:
            print("Missing required AWS credentials in environment variables")

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
        """Load data from either local file or S3"""
        # Parse the filepath to check if it's an S3 URL
        if filepath.startswith('s3://'):
            if not self.s3_client:
                raise ValueError("AWS credentials not found in environment variables")
            
            # Parse S3 URL
            parsed = urlparse(filepath)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            try:
                # Get object from S3
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                data = json.loads(response['Body'].read().decode('utf-8'))
            except Exception as e:
                raise Exception(f"Error loading data from S3: {str(e)}")
        else:
            # Load from local file
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                raise Exception(f"Error loading local file: {str(e)}")

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
        print(f"Number of categories: {self.condition_dim}")

        normalized = self.scaler.fit_transform(df[['amount', 'timestamp']])
        X = normalized
        C = category_dummies.values

        self.output_dim = X.shape[1]  # For critic input
        print(f"Input shape: {X.shape}, Condition shape: {C.shape}")
        
        # Create tensors on CPU first
        tensor_X = torch.FloatTensor(X)
        tensor_C = torch.FloatTensor(C)
        
        # Move tensors to the correct device
        tensor_X = tensor_X.to(self.device)
        tensor_C = tensor_C.to(self.device)
        
        # Synchronize if using MPS device
        if self.device.type == 'mps':
            torch.mps.synchronize()
        
        return tensor_X, tensor_C

    def generate_transaction_details(self, category, amount):
        """Generate transaction details based on learned patterns"""
        import random
        
        merchants = self.merchant_by_category[category]
        descriptions = self.descriptions_by_category[category]
        
        # Determine if this is an income transaction
        is_income = category in self.income_categories
        
        # Adjust amount sign based on transaction type
        amount = abs(amount)  # Make sure amount is positive first
        if not is_income:
            amount = -amount  # Make expenses negative
        
        if not merchants or not any(merchants):  
            merchant = ""
            description = random.choice(descriptions) if descriptions else f"Transaction - {category}"
        else:
            merchant = random.choice(merchants)
            # use fixed price if merchant has one
            if merchant in self.fixed_price_merchants:
                amount = self.fixed_price_merchants[merchant]
                if not is_income:
                    amount = -amount
            
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
        base_timestamp = int(datetime.now().timestamp())
        
        # Create a set to track used transaction IDs
        used_transaction_ids = set()
        
        def generate_unique_transaction_id():
            """Generate a unique transaction ID with timestamp and random component"""
            while True:
                # Combine timestamp with a random number for uniqueness
                random_component = np.random.randint(1000000, 9999999)
                tx_id = f"TX{base_timestamp}{random_component}"
                if tx_id not in used_transaction_ids:
                    used_transaction_ids.add(tx_id)
                    return tx_id
        
        # Group transactions by month to track salary, subscriptions, and utilities
        monthly_salary = {}
        monthly_subscriptions = {}  # Track subscription payments by month
        monthly_utilities = {}  # Track utility payments by month
        
        # First pass: Create all transactions and track salary/subscriptions/utilities
        temp_transactions = []
        for i in range(len(generated_data)):
            category_idx = np.argmax(categorical_data[i])
            category = self.category_columns[category_idx].replace('category_', '')
            amount = denormalized_data[i, 0]
            
            timestamp = int(denormalized_data[i, 1])
            booking_date = datetime.fromtimestamp(timestamp)
            month_key = f"{booking_date.year}-{booking_date.month}"
            
            # For salary transactions, track them by month
            if category == "Salary":
                if month_key not in monthly_salary:
                    monthly_salary[month_key] = []
                monthly_salary[month_key].append((i, amount, booking_date))
                continue
            
            # For subscription transactions, track them by month
            if category == "Subscriptions":
                if month_key not in monthly_subscriptions:
                    monthly_subscriptions[month_key] = {}
                
                # Choose a subscription merchant that hasn't been used this month
                available_merchants = [
                    m for m in self.subscription_merchants.keys()
                    if m not in monthly_subscriptions[month_key]
                ]
                
                if available_merchants:
                    merchant = random.choice(available_merchants)
                    monthly_subscriptions[month_key][merchant] = (i, booking_date)
                    
                    # Create subscription transaction with fixed amount
                    transaction = {
                        "transactionId": generate_unique_transaction_id(),
                        "bookingDateTime": booking_date.strftime("%d/%m/%Y %H:%M:%S"),
                        "valueDateTime": booking_date.strftime("%d/%m/%Y %H:%M:%S"),
                        "transactionAmount": {
                            "amount": f"{-self.subscription_merchants[merchant]:.2f}",
                            "currency": "EUR"
                        },
                        "creditorName": merchant,
                        "creditorAccount": {
                            "iban": f"DE{np.random.randint(1000000000, 9999999999)}"
                        },
                        "debtorName": f"Person{np.random.randint(1000, 9999)}",
                        "debtorAccount": {
                            "iban": f"DE{np.random.randint(1000000000, 9999999999)}"
                        },
                        "remittanceInformationUnstructured": f"Monthly Subscription - {merchant}",
                        "category": "Subscriptions"
                    }
                    temp_transactions.append(transaction)
                continue

            # For utility transactions, track them by month
            if category == "Utilities":
                if month_key not in monthly_utilities:
                    monthly_utilities[month_key] = {}
                
                # Choose a utility merchant that hasn't been used this month
                available_merchants = [
                    m for m in self.utility_merchants.keys()
                    if m not in monthly_utilities[month_key]
                ]
                
                if available_merchants:
                    merchant = random.choice(available_merchants)
                    monthly_utilities[month_key][merchant] = (i, booking_date)
                    
                    # Create utility transaction with fixed amount
                    transaction = {
                        "transactionId": generate_unique_transaction_id(),
                        "bookingDateTime": booking_date.strftime("%d/%m/%Y %H:%M:%S"),
                        "valueDateTime": booking_date.strftime("%d/%m/%Y %H:%M:%S"),
                        "transactionAmount": {
                            "amount": f"{-self.utility_merchants[merchant]:.2f}",
                            "currency": "EUR"
                        },
                        "creditorName": merchant,
                        "creditorAccount": {
                            "iban": f"DE{np.random.randint(1000000000, 9999999999)}"
                        },
                        "debtorName": f"Person{np.random.randint(1000, 9999)}",
                        "debtorAccount": {
                            "iban": f"DE{np.random.randint(1000000000, 9999999999)}"
                        },
                        "remittanceInformationUnstructured": f"Monthly Utility Bill - {merchant}",
                        "category": "Utilities"
                    }
                    temp_transactions.append(transaction)
                continue
            
            merchant_name, description, amount = self.generate_transaction_details(category, amount)
            
            # Format amount with sign for display
            amount_str = f"{amount:.2f}"
            
            # Format date in dd/mm/yyyy format
            formatted_date = booking_date.strftime("%d/%m/%Y %H:%M:%S")
            
            transaction = {
                "transactionId": generate_unique_transaction_id(),
                "bookingDateTime": formatted_date,
                "valueDateTime": formatted_date,
                "transactionAmount": {
                    "amount": amount_str,
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
            temp_transactions.append(transaction)
        
        # Second pass: Add exactly one salary transaction per month
        for month, salary_transactions in monthly_salary.items():
            if salary_transactions:
                # Choose the salary transaction with the date closest to the start of the month
                salary_transactions.sort(key=lambda x: x[2].day)
                chosen_salary = salary_transactions[0]
                
                merchant_name, description, amount = self.generate_transaction_details("Salary", chosen_salary[1])
                
                # Format salary date in dd/mm/yyyy format
                salary_date = chosen_salary[2].replace(day=1)  # Always on the 1st
                formatted_salary_date = salary_date.strftime("%d/%m/%Y %H:%M:%S")
                
                # Create the salary transaction
                transaction = {
                    "transactionId": generate_unique_transaction_id(),
                    "bookingDateTime": formatted_salary_date,
                    "valueDateTime": formatted_salary_date,
                    "transactionAmount": {
                        "amount": f"{amount:.2f}",
                        "currency": "EUR"
                    },
                    "creditorName": "Employer Corp.",  # Fixed employer name for consistency
                    "creditorAccount": {
                        "iban": f"DE{np.random.randint(1000000000, 9999999999)}"
                    },
                    "debtorName": f"Person{np.random.randint(1000, 9999)}",
                    "debtorAccount": {
                        "iban": f"DE{np.random.randint(1000000000, 9999999999)}"
                    },
                    "remittanceInformationUnstructured": "Monthly Salary Payment",
                    "category": "Salary"
                }
                temp_transactions.append(transaction)
        
        # Sort all transactions by date
        transactions = sorted(temp_transactions, key=lambda x: datetime.strptime(x['bookingDateTime'], "%d/%m/%Y %H:%M:%S"))
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

    def upload_dataset_to_s3(self, data: dict, username: str, dataset_name: str) -> str:
        """
        Upload a dataset to S3 in the user_datasets directory.
        Returns the S3 URL of the uploaded dataset.
        """
        if not self.s3_client:
            raise ValueError("AWS credentials not found in environment variables")

        # Validate the dataset structure
        required_fields = ['transactionAmount', 'bookingDateTime', 'category']
        for transaction in data:
            if not all(field in transaction for field in required_fields):
                raise ValueError("Invalid dataset structure. Missing required fields.")

        # Create a safe filename from the dataset name
        safe_filename = "".join(c for c in dataset_name if c.isalnum() or c in ('-', '_')).lower()
        key = f"user_datasets/{username}/{safe_filename}.json"
        bucket = "synthetic-personas-training-datasets"

        try:
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(data),
                ContentType='application/json'
            )
            return f"s3://{bucket}/{key}"
        except Exception as e:
            raise Exception(f"Error uploading to S3: {str(e)}")

    def validate_custom_distribution(self, distribution: dict) -> bool:
        """
        Validate a custom category distribution.
        Returns True if valid, raises ValueError if invalid.
        """
        if not distribution:
            raise ValueError("Distribution cannot be empty")

        # Validate values are numbers and sum to 1
        try:
            total = sum(float(val) for val in distribution.values())
            if not (0.99 <= total <= 1.01):  # Allow small rounding errors
                raise ValueError(f"Distribution must sum to 100% (got {total * 100}%)")
        except (TypeError, ValueError):
            raise ValueError("Distribution values must be numbers between 0 and 1")

        return True

    def transaction_to_tensor(self, tx):
        """Convert a transaction to a tensor format suitable for the model."""
        # Extract numerical features
        try:
            if isinstance(tx, dict):
                amount = float(tx['transactionAmount']['amount'])
            else:
                amount = float(tx.amount)
        except (KeyError, AttributeError):
            print(f"Invalid transaction format: {tx}")
            raise ValueError("Invalid transaction format - missing amount field")

        # Extract temporal features
        try:
            if isinstance(tx, dict):
                booking_date = pd.to_datetime(tx['bookingDateTime'], dayfirst=True)  # Add dayfirst=True
            else:
                booking_date = pd.to_datetime(tx.booking_date_time)
        except (KeyError, AttributeError):
            print(f"Invalid transaction format: {tx}")
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
            print(f"Invalid transaction format: {tx}")
            raise ValueError("Invalid transaction format - missing category field")

        category_vector = np.zeros(len(self.category_columns))
        try:
            if category.startswith('category_'):
                category_name = category
            else:
                category_name = f"category_{category}"
            category_idx = list(self.category_columns).index(category_name)
            category_vector[category_idx] = 1
        except ValueError:
            # If category not found, use a random category
            category_vector[random.randrange(len(self.category_columns))] = 1
        
        # Combine features
        features = np.array([amount, day_of_month, day_of_week])
        
        # Normalize numerical features
        if not hasattr(self, '_feature_scaler'):
            self._feature_scaler = MinMaxScaler()
            self._feature_scaler.fit(np.array([[0, 0, 0], [10000, 1, 1]]))
        features = self._feature_scaler.transform(features.reshape(1, -1))[0]
        
        # Create tensors on CPU first
        input_tensor = torch.FloatTensor(features[:2])  # Only amount and day_of_month
        condition_tensor = torch.FloatTensor(category_vector)  # Shape: [n_categories]
        
        # Move tensors to the correct device
        input_tensor = input_tensor.to(self.device)
        condition_tensor = condition_tensor.to(self.device)
        
        # Synchronize if using MPS device
        if self.device.type == 'mps':
            torch.mps.synchronize()
        
        return input_tensor, condition_tensor 