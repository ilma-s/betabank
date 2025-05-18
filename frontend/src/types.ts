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