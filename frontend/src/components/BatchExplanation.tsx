import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Info, Edit2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import axios from "axios";
import { TemporalPattern } from "@/types";

interface BatchExplanationData {
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
      distribution: Record<
        string,
        {
          count: number;
          percentage: number;
          average_amount: number;
        }
      >;
      transitions: Array<{
        from: string;
        to: string;
        count: number;
        probability: number;
      }>;
      temporal: Array<{
        category: string;
        patterns: TemporalPattern[];
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

interface PatternAnnotation {
  type: string;
  description: string;
  confidence: number;
}

interface Props extends BatchExplanationData {
  token: string;
}

export function BatchExplanation(props: Props) {
  const { token, ...data } = props;
  const [showAnnotationDialog, setShowAnnotationDialog] = useState(false);
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);
  const [annotationText, setAnnotationText] = useState("");
  const [userAnnotations, setUserAnnotations] = useState<
    Record<string, PatternAnnotation>
  >({});

  const handleAnnotationSubmit = async () => {
    if (!selectedPattern) return;

    try {
      await axios.post(
        `http://localhost:8000/batches/${data.batch_id}/explanation/annotations`,
        {
          pattern_name: selectedPattern,
          description: annotationText,
          confidence: 1.0,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );

      setUserAnnotations((prev) => ({
        ...prev,
        [selectedPattern]: {
          type: "user",
          description: annotationText,
          confidence: 1.0,
        },
      }));
      setShowAnnotationDialog(false);
      setAnnotationText("");
    } catch (error) {
      console.error(
        "Error submitting annotation:",
        error instanceof Error ? error.message : "Unknown error"
      );
    }
  };

  if (!data) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Batch Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>No analysis data available.</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const categoryData = Object.entries(
    data.distribution_explanation?.category_distribution?.distribution || {}
  )
    .map(([category, stats]) => ({
      category,
      percentage: stats.percentage * 100,
      averageAmount: stats.average_amount,
      count: stats.count,
    }))
    .sort((a, b) => b.percentage - a.percentage);

  const timeClusterData = (data.temporal_patterns?.time_clusters || [])
    .map((cluster) => ({
      hour: cluster.hour,
      count: cluster.count,
      density: cluster.density * 100,
    }))
    .sort((a, b) => b.count - a.count);

  return (
    <div className="space-y-4">
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Batch Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert className="mb-4">
            <Info className="h-4 w-4" />
            <AlertDescription>{data.summary_text}</AlertDescription>
          </Alert>

          <Tabs defaultValue="distribution" className="w-full">
            <TabsList>
              <TabsTrigger value="distribution">Distribution</TabsTrigger>
              <TabsTrigger value="temporal">Temporal</TabsTrigger>
              <TabsTrigger value="amounts">Amounts</TabsTrigger>
              <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
              <TabsTrigger value="annotations">Annotations</TabsTrigger>
            </TabsList>

            <TabsContent value="distribution">
              <Card>
                <CardHeader>
                  <CardTitle>Category Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  {categoryData.length > 0 ? (
                    <>
                      <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={categoryData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="category" />
                            <YAxis />
                            <Tooltip />
                            <Bar
                              dataKey="percentage"
                              fill="#8884d8"
                              name="Percentage"
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <ScrollArea className="h-[200px] mt-4">
                        <div className="space-y-2">
                          {data.distribution_explanation?.category_distribution?.transitions?.map(
                            (transition, idx) => (
                              <div
                                key={idx}
                                className="flex items-center gap-2"
                              >
                                <Badge variant="outline">
                                  {transition.from}
                                </Badge>
                                →
                                <Badge variant="outline">{transition.to}</Badge>
                                <span className="text-sm text-muted-foreground">
                                  ({(transition.probability * 100).toFixed(1)}%
                                  probability)
                                </span>
                              </div>
                            )
                          )}
                        </div>
                      </ScrollArea>
                    </>
                  ) : (
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertDescription>
                        No category distribution data available.
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="temporal">
              <Card>
                <CardHeader>
                  <CardTitle>Temporal Patterns</CardTitle>
                </CardHeader>
                <CardContent>
                  {data.temporal_patterns?.regular_intervals?.map(
                    (interval, idx) => (
                      <div key={idx} className="mb-2">
                        <Badge variant="secondary">Regular Interval</Badge>
                        <p className="mt-1">
                          Transactions occur every{" "}
                          {interval.interval_days.toFixed(1)} days (Confidence:{" "}
                          {(interval.confidence * 100).toFixed(1)}%)
                        </p>
                      </div>
                    )
                  )}

                  {data.temporal_patterns?.periodic_transactions?.map(
                    (pattern, idx) => (
                      <div key={idx} className="mb-2">
                        <Badge variant="secondary">Monthly Pattern</Badge>
                        <p className="mt-1">
                          {pattern.count} transactions on day{" "}
                          {pattern.day_of_month}
                          (Confidence: {(pattern.confidence * 100).toFixed(1)}%)
                        </p>
                      </div>
                    )
                  )}

                  {timeClusterData.length > 0 && (
                    <div className="h-[200px] w-full mt-4">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={timeClusterData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="hour" />
                          <YAxis />
                          <Tooltip />
                          <Bar
                            dataKey="count"
                            fill="#82ca9d"
                            name="Transaction Count"
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {!data.temporal_patterns?.regular_intervals?.length &&
                    !data.temporal_patterns?.periodic_transactions?.length &&
                    !timeClusterData.length && (
                      <Alert>
                        <Info className="h-4 w-4" />
                        <AlertDescription>
                          No temporal patterns detected.
                        </AlertDescription>
                      </Alert>
                    )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="amounts">
              <Card>
                <CardHeader>
                  <CardTitle>Amount Patterns</CardTitle>
                </CardHeader>
                <CardContent>
                  {data.amount_patterns?.fixed_amounts?.length ||
                  data.amount_patterns?.amount_ranges?.length ? (
                    <div className="space-y-4">
                      {data.amount_patterns.fixed_amounts?.length > 0 && (
                        <div>
                          <h4 className="font-medium mb-2">Common Amounts</h4>
                          <div className="space-y-2">
                            {data.amount_patterns.fixed_amounts.map(
                              (amount, idx) => (
                                <div
                                  key={idx}
                                  className="flex items-center gap-2"
                                >
                                  <Badge variant="outline">
                                    ${amount.amount.toFixed(2)}
                                  </Badge>
                                  <span className="text-sm text-muted-foreground">
                                    ({amount.frequency} times,{" "}
                                    {(amount.confidence * 100).toFixed(1)}%
                                    confidence)
                                  </span>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      )}

                      {data.amount_patterns.amount_ranges?.length > 0 && (
                        <div>
                          <h4 className="font-medium mb-2">Amount Ranges</h4>
                          <div className="space-y-2">
                            {data.amount_patterns.amount_ranges.map(
                              (range, idx) => (
                                <div key={idx}>
                                  <Badge variant="outline">
                                    ${range.mean.toFixed(2)} ± $
                                    {range.std.toFixed(2)}
                                  </Badge>
                                  <p className="text-sm text-muted-foreground mt-1">
                                    Range: ${range.range[0].toFixed(2)} - $
                                    {range.range[1].toFixed(2)}(
                                    {(range.weight * 100).toFixed(1)}% of
                                    transactions)
                                  </p>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertDescription>
                        No amount patterns detected.
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="anomalies">
              <Card>
                <CardHeader>
                  <CardTitle>Anomalies</CardTitle>
                </CardHeader>
                <CardContent>
                  {data.anomalies?.length > 0 ? (
                    <ScrollArea className="h-[300px]">
                      <div className="space-y-4">
                        {data.anomalies.map((anomaly, idx) => (
                          <div key={idx} className="p-4 border rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                              <Badge variant="destructive">Anomaly</Badge>
                              <span className="font-medium">
                                Transaction {anomaly.transaction_id}
                              </span>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              Amount: ${anomaly.amount.toFixed(2)}
                              <br />
                              Expected Range: $
                              {anomaly.expected_range[0].toFixed(2)} - $
                              {anomaly.expected_range[1].toFixed(2)}
                              <br />
                              Reason: {anomaly.reason}
                            </p>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  ) : (
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertDescription>
                        No anomalies detected.
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="annotations">
              <Card>
                <CardHeader>
                  <CardTitle>Pattern Annotations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(userAnnotations).map(
                      ([pattern, annotation]) => (
                        <div key={pattern} className="p-4 border rounded-lg">
                          <div className="flex items-center justify-between">
                            <Badge>{pattern}</Badge>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setSelectedPattern(pattern);
                                setAnnotationText(annotation.description);
                                setShowAnnotationDialog(true);
                              }}
                            >
                              <Edit2 className="h-4 w-4" />
                            </Button>
                          </div>
                          <p className="mt-2 text-sm">
                            {annotation.description}
                          </p>
                        </div>
                      )
                    )}

                    <Button
                      onClick={() => {
                        setSelectedPattern(null);
                        setAnnotationText("");
                        setShowAnnotationDialog(true);
                      }}
                      className="w-full"
                    >
                      Add New Annotation
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <Dialog
        open={showAnnotationDialog}
        onOpenChange={setShowAnnotationDialog}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {selectedPattern ? "Edit Annotation" : "Add New Annotation"}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            {!selectedPattern && (
              <div>
                <label className="text-sm font-medium">Pattern Name</label>
                <Input
                  value={selectedPattern || ""}
                  onChange={(e) => setSelectedPattern(e.target.value)}
                  placeholder="e.g., Monthly Rent Pattern"
                />
              </div>
            )}
            <div>
              <label className="text-sm font-medium">Description</label>
              <Input
                value={annotationText}
                onChange={(e) => setAnnotationText(e.target.value)}
                placeholder="Describe the pattern you've observed..."
              />
            </div>
            <Button onClick={handleAnnotationSubmit} className="w-full">
              {selectedPattern ? "Update Annotation" : "Add Annotation"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
