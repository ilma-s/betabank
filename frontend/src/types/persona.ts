export interface PersonaWithDataset {
  id: number;
  name: string;
  description: string;
  dataset: Array<{
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
  }>;
} 