import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff8042'];

interface TransactionChartsProps {
  transactions: any[];
  personaType: string;
}

export function TransactionCharts({ transactions, personaType }: TransactionChartsProps) {
  // Calculate category distribution
  const categoryDistribution = transactions.reduce((acc: any, tx) => {
    const category = tx.category;
    if (!acc[category]) {
      acc[category] = { count: 0, amount: 0 };
    }
    acc[category].count++;
    acc[category].amount += parseFloat(tx.transactionAmount.amount);
    return acc;
  }, {});

  // Convert to array format for charts
  const pieData = Object.entries(categoryDistribution).map(([category, data]: [string, any]) => ({
    name: category,
    value: data.count,
    amount: data.amount
  }));

  // Get critical category based on persona
  const getCriticalCategory = () => {
    switch (personaType.toLowerCase()) {
      case 'gambling addict':
        return 'Gambling';
      case 'shopping addict':
        return 'Shopping';
      case 'crypto enthusiast':
        return 'Crypto';
      case 'money mule':
        return 'Transfer';
      default:
        return null;
    }
  };

  const criticalCategory = getCriticalCategory();
  const criticalData = criticalCategory ? 
    transactions
      .filter(tx => tx.category === criticalCategory)
      .map(tx => ({
        date: new Date(tx.bookingDateTime).toLocaleDateString(),
        amount: parseFloat(tx.transactionAmount.amount)
      }))
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    : [];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Pie Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-[#261436]">Category Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={false}
                  outerRadius={90}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value: number, name: string, props: any) => [
                    `${value} transactions (${props.payload.amount.toFixed(2)} EUR)`,
                    name
                  ]}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Bar Chart for Critical Category */}
      {criticalCategory && (
        <Card>
          <CardHeader>
            <CardTitle className="text-[#261436]">{criticalCategory} Transactions Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            {criticalData.length > 0 ? (
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={criticalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="date" 
                      angle={-45}
                      textAnchor="end"
                      height={60}
                      interval={Math.ceil(criticalData.length / 10)}
                    />
                    <YAxis />
                    <Tooltip 
                      formatter={(value: number) => [`${value.toFixed(2)} EUR`, 'Amount']}
                    />
                    <Bar dataKey="amount" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-[#261436] opacity-70">
                No {criticalCategory.toLowerCase()} transactions in this batch.
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
} 