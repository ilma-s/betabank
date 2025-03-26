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
  id: string;
  name: string;
  description: string;
}

export interface TransactionBatch {
  id: number;
  name: string;
  persona_id: number;
  persona_name: string;
  created_at: string;
  transaction_count: number;
  preview?: Transaction[];
} 