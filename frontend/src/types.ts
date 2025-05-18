export interface Transaction {
  transactionId: string;
  bookingDateTime: string;
  valueDateTime: string;
  transactionAmount: {
    amount: string;
    currency: string;
  };
  creditorName: string;
  creditorAccount: {
    iban: string;
  };
  debtorName: string;
  debtorAccount: {
    iban: string;
  };
  remittanceInformationUnstructured: string;
  category: string;
  edited?: boolean;
}

export interface TransactionExplanationData {
  transaction_id: string;
  feature_importance: {
    [key: string]: number;
  };
  applied_patterns: {
    [key: string]: {
      type: string;
      value: any;
    };
  };
  explanation_text: string;
  confidence_score: number;
  meta_info?: {
    generated_at: string;
    version: string;
    [key: string]: any;
  };
}

export interface BatchExplanationData {
  batch_id: number;
  distribution_explanation: {
    amount_distribution: {
      fixed_amounts: Array<{
        amount: number;
        frequency: number;
        confidence: number;
      }>;
      amount_ranges: Array<{
        mean: number;
        std: number;
        range: [number, number];
        weight: number;
      }>;
    };
    category_distribution: {
      distribution: Record<string, {
        count: number;
        percentage: number;
        average_amount: number;
      }>;
      transitions: Array<{
        from: string;
        to: string;
        count: number;
        probability: number;
      }>;
      temporal: Array<{
        category: string;
        patterns: any;
      }>;
    };
    transaction_count: number;
  };
  temporal_patterns: {
    regular_intervals: Array<{
      interval_days: number;
      confidence: number;
    }>;
    periodic_transactions: Array<{
      day_of_month: number;
      count: number;
      confidence: number;
    }>;
    time_clusters: Array<{
      hour: number;
      density: number;
      count: number;
    }>;
  };
  amount_patterns: {
    fixed_amounts: Array<{
      amount: number;
      frequency: number;
      confidence: number;
    }>;
    amount_ranges: Array<{
      mean: number;
      std: number;
      range: [number, number];
      weight: number;
    }>;
  };
  anomalies: Array<{
    transaction_id: string;
    amount: number;
    reason: string;
    expected_range: [number, number];
  }>;
  summary_text: string;
}

export interface Persona {
  id: number;
  name: string;
  description: string;
  config_json?: {
    dataset_path: string;
    custom_distribution?: Record<string, number>;
    use_for_training?: boolean;
  };
}

export interface TransactionBatch {
  id: number;
  name: string;
  persona_id: number;
  persona_name: string;
  created_at: string;
  transaction_count: number;
  preview: any;
  months: number;
  distribution?: Record<string, number>;
} 