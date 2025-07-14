from sqlalchemy.orm import Session
from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats
from .database_models import (
    Transaction, TransactionExplanation, BatchExplanation,
    PatternLibrary, TransactionBatch
)

class ExplanationService:
    def __init__(self, db: Session):
        self.db = db
        self.patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict[str, List[Dict]]:
        """Load patterns from the pattern library grouped by type."""
        patterns = self.db.query(PatternLibrary).all()
        pattern_dict = {
            "temporal": [],
            "amount": [],
            "category": [],
            "distribution": []
        }
        for pattern in patterns:
            pattern_dict[pattern.pattern_type].append({
                "id": pattern.id,
                "name": pattern.name,
                "rules": pattern.rules,
                "description": pattern.description
            })
        return pattern_dict
    
    def detect_temporal_patterns(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Enhanced temporal pattern detection."""
        if not transactions:
            return {"regular_intervals": [], "periodic_transactions": [], "time_clusters": []}

        # Sort transactions by date
        sorted_txs = sorted(transactions, key=lambda tx: tx.booking_date_time)
        dates = [tx.booking_date_time for tx in sorted_txs]
        
        patterns = {
            "regular_intervals": [],
            "periodic_transactions": [],
            "time_clusters": []
        }
        
        # Detect regular intervals
        if len(dates) > 1:
            intervals = np.diff([d.timestamp() for d in dates])
            mean_interval = np.mean(intervals) / (24 * 3600)  # Convert to days
            std_interval = np.std(intervals) / (24 * 3600)
            
            if std_interval / mean_interval < 0.2:  # Check if intervals are regular
                patterns["regular_intervals"].append({
                    "interval_days": mean_interval,
                    "confidence": 1 - (std_interval / mean_interval)
                })

        # Detect periodic transactions (monthly patterns)
        day_counts = defaultdict(int)
        for date in dates:
            day_counts[date.day] = day_counts[date.day] + 1

        total_txs = len(dates)
        for day, count in day_counts.items():
            if count >= 2:  # At least 2 occurrences
                confidence = count / (total_txs / 30)  # Normalize by expected frequency
                if confidence > 0.5:  # Only include strong patterns
                    patterns["periodic_transactions"].append({
                        "day_of_month": day,
                        "count": count,
                        "confidence": min(confidence, 1.0)
                    })

        # Detect time clusters
        hours = [d.hour for d in dates]
        if hours and len(set(hours)) > 1:
            kde = stats.gaussian_kde(hours)
            hour_range = np.arange(24)
            density = kde(hour_range)
            peaks = []
            
            for i in range(1, 23):
                if density[i] > density[i-1] and density[i] > density[i+1]:
                    peaks.append((i, density[i]))
            
            for hour, density_value in peaks:
                patterns["time_clusters"].append({
                    "hour": hour,
                    "density": float(density_value),
                    "count": sum(1 for h in hours if abs(h - hour) <= 1)
                })

        return patterns
    
    def detect_amount_patterns(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Enhanced amount pattern detection."""
        if not transactions:
            return {"fixed_amounts": [], "amount_ranges": []}

        amounts = [tx.amount for tx in transactions]
        patterns = {
            "fixed_amounts": [],
            "amount_ranges": []
        }

        # Detect fixed amounts (common values)
        amount_counts = defaultdict(int)
        for amount in amounts:
            # Round to 2 decimal places for comparison
            rounded = round(amount, 2)
            amount_counts[rounded] += 1

        total_txs = len(amounts)
        for amount, count in amount_counts.items():
            if count >= 2:  # At least 2 occurrences
                frequency = count / total_txs
                if frequency >= 0.1:  # 10% or more of transactions
                    patterns["fixed_amounts"].append({
                        "amount": amount,
                        "frequency": count,
                        "confidence": frequency
                    })

        # Detect amount ranges using Gaussian Mixture Model
        if len(amounts) >= 5:  # Need enough data points
            from sklearn.mixture import GaussianMixture
            
            X = np.array(amounts).reshape(-1, 1)
            n_components = min(3, len(amounts) // 5)  # Maximum 3 clusters
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(X)
            
            for i in range(n_components):
                mean = float(gmm.means_[i][0])
                std = float(np.sqrt(gmm.covariances_[i][0]))
                weight = float(gmm.weights_[i])
                
                if weight >= 0.1:  # Only include significant clusters
                    patterns["amount_ranges"].append({
                        "mean": mean,
                        "std": std,
                        "range": [mean - 2*std, mean + 2*std],
                        "weight": weight
                    })

        return patterns
    
    def detect_category_patterns(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect patterns in category distribution and transitions."""
        if not transactions:
            return {"distribution": {}, "transitions": [], "temporal": []}

        patterns = {
            "distribution": {},
            "transitions": [],
            "temporal": []
        }

        # Analyze category distribution
        category_counts = defaultdict(int)
        total_txs = len(transactions)
        
        for tx in transactions:
            category_counts[tx.category] += 1

        for category, count in category_counts.items():
            patterns["distribution"][category] = {
                "count": count,
                "percentage": count / total_txs,
                "average_amount": np.mean([tx.amount for tx in transactions if tx.category == category])
            }

        # Analyze category transitions
        sorted_txs = sorted(transactions, key=lambda tx: tx.booking_date_time)
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(sorted_txs) - 1):
            curr_cat = sorted_txs[i].category
            next_cat = sorted_txs[i + 1].category
            transitions[curr_cat][next_cat] += 1

        # Find significant transitions
        for from_cat, to_cats in transitions.items():
            total = sum(to_cats.values())
            for to_cat, count in to_cats.items():
                probability = count / total
                if probability >= 0.3:  # 30% or higher transition probability
                    patterns["transitions"].append({
                        "from": from_cat,
                        "to": to_cat,
                        "count": count,
                        "probability": probability
                    })

        # Analyze temporal category patterns
        for category in category_counts:
            cat_txs = [tx for tx in transactions if tx.category == category]
            if len(cat_txs) >= 3:
                temporal_pattern = self.detect_temporal_patterns(cat_txs)
                if any(temporal_pattern.values()):
                    patterns["temporal"].append({
                        "category": category,
                        "patterns": temporal_pattern
                    })

        return patterns
    
    def generate_transaction_explanation(
        self, 
        transaction: Transaction, 
        feature_importance: Dict[str, float]
    ) -> TransactionExplanation:
        """Generate enhanced explanation for a single transaction."""
        patterns = {}
        explanation_parts = []
        
        # Add feature importance explanation
        important_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        explanation_parts.append(
            f"This transaction was primarily influenced by: "
            f"{', '.join(f'{f[0]} ({f[1]:.2%})' for f in important_features)}"
        )
        
        # Check temporal patterns
        if transaction.booking_date_time.day in [1, 15, 30]:
            patterns["temporal"] = {
                "type": "fixed_date",
                "value": transaction.booking_date_time.day
            }
            explanation_parts.append(
                f"Occurs on day {transaction.booking_date_time.day} of the month"
            )
        
        # Check amount patterns
        if transaction.amount in [10, 20, 50, 100, 200, 500, 1000]:
            patterns["amount"] = {
                "type": "round_amount",
                "value": transaction.amount
            }
            explanation_parts.append(f"Uses a round amount of {transaction.amount}")
        
        # Check for similar transactions
        similar_txs = (
            self.db.query(Transaction)
            .filter(
                Transaction.batch_id == transaction.batch_id,
                Transaction.category == transaction.category,
                Transaction.transaction_id != transaction.transaction_id
            )
            .all()
        )
        
        if similar_txs:
            similar_amounts = [tx.amount for tx in similar_txs]
            mean_amount = np.mean(similar_amounts)
            std_amount = np.std(similar_amounts)
            
            if abs(transaction.amount - mean_amount) > 2 * std_amount:
                patterns["anomaly"] = {
                    "type": "amount_outlier",
                    "expected_range": [mean_amount - 2*std_amount, mean_amount + 2*std_amount],
                    "actual": transaction.amount
                }
                explanation_parts.append(
                    f"This amount (${transaction.amount:.2f}) is unusual for this category "
                    f"(typically ${mean_amount:.2f} ± ${2*std_amount:.2f})"
                )
        
        explanation_text = " ".join(explanation_parts)
        confidence_score = np.mean(list(feature_importance.values()))
        
        explanation = TransactionExplanation(
            transaction_id=transaction.transaction_id,
            batch_id=transaction.batch_id,
            feature_importance=feature_importance,
            applied_patterns=patterns,
            explanation_text=explanation_text,
            confidence_score=confidence_score,
            meta_info={
                "generated_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "similar_transactions_count": len(similar_txs)
            }
        )
        
        return explanation
    
    def generate_batch_explanation(
        self, 
        batch: TransactionBatch,
        transactions: List[Transaction]
    ) -> BatchExplanation:
        """Generate enhanced batch explanation."""
        temporal_patterns = self.detect_temporal_patterns(transactions)
        amount_patterns = self.detect_amount_patterns(transactions)
        category_patterns = self.detect_category_patterns(transactions)
        
        # Detect anomalies
        amounts = [tx.amount for tx in transactions]
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        anomalies = []
        
        for tx in transactions:
            if abs(tx.amount - mean_amount) > 3 * std_amount:
                anomalies.append({
                    "transaction_id": tx.transaction_id,
                    "amount": tx.amount,
                    "reason": "amount_outlier",
                    "expected_range": [mean_amount - 3*std_amount, mean_amount + 3*std_amount]
                })
        
        # Generate summary text
        summary_parts = []
        
        # Add temporal pattern summary
        if temporal_patterns["regular_intervals"]:
            pattern = temporal_patterns["regular_intervals"][0]
            summary_parts.append(
                f"Transactions occur regularly every {pattern['interval_days']:.1f} days"
            )
        
        if temporal_patterns["periodic_transactions"]:
            monthly_patterns = sorted(
                temporal_patterns["periodic_transactions"],
                key=lambda x: x["count"],
                reverse=True
            )[:2]
            days_str = ", ".join(f"day {p['day_of_month']}" for p in monthly_patterns)
            summary_parts.append(f"Regular monthly transactions on {days_str}")
        
        # Add amount pattern summary
        if amount_patterns["fixed_amounts"]:
            common_amounts = sorted(
                amount_patterns["fixed_amounts"],
                key=lambda x: x["frequency"],
                reverse=True
            )[:3]
            amounts_str = ", ".join(f"${a['amount']:.2f}" for a in common_amounts)
            summary_parts.append(f"Common transaction amounts: {amounts_str}")
        
        # Add category pattern summary
        top_categories = sorted(
            category_patterns["distribution"].items(),
            key=lambda x: x[1]["percentage"],
            reverse=True
        )[:3]
        categories_str = ", ".join(
            f"{cat} ({stats['percentage']:.1%})"
            for cat, stats in top_categories
        )
        summary_parts.append(f"Most frequent categories: {categories_str}")
        
        if anomalies:
            summary_parts.append(
                f"Found {len(anomalies)} unusual transactions that deviate "
                "significantly from the normal pattern"
            )
        
        explanation = BatchExplanation(
            batch_id=batch.id,
            distribution_explanation={
                "amount_distribution": amount_patterns,
                "category_distribution": category_patterns,
                "transaction_count": len(transactions)
            },
            temporal_patterns=temporal_patterns,
            amount_patterns=amount_patterns,
            anomalies=anomalies,
            summary_text=" ".join(summary_parts)
        )
        
        return explanation
    
    def process_batch(
        self, 
        batch: TransactionBatch,
        feature_importances: List[Dict[str, float]]
    ) -> None:
        """Process a batch of transactions and generate all explanations."""
        transactions = self.db.query(Transaction).filter(
            Transaction.batch_id == batch.id
        ).all()
        
        # Generate individual transaction explanations
        for tx, importance in zip(transactions, feature_importances):
            explanation = self.generate_transaction_explanation(tx, importance)
            self.db.add(explanation)
        
        # Generate batch-level explanation
        batch_explanation = self.generate_batch_explanation(batch, transactions)
        self.db.add(batch_explanation)
        
        self.db.commit() 