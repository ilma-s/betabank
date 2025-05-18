import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { TransactionExplanationData } from "@/types";

interface TransactionExplanationProps
  extends Omit<TransactionExplanationData, "transaction_id"> {
  transactionId: string;
}

export function TransactionExplanation({
  feature_importance: featureImportance,
  applied_patterns: appliedPatterns,
  explanation_text: explanationText,
  confidence_score: confidenceScore,
  meta_info,
}: TransactionExplanationProps) {
  const sortedFeatures = Object.entries(featureImportance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Transaction Explanation</CardTitle>
        <CardDescription>
          Understanding the factors behind this transaction
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="text-lg font-medium">{explanationText}</div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Confidence Score</span>
              <span>{(confidenceScore * 100).toFixed(1)}%</span>
            </div>
            <Progress value={confidenceScore * 100} />
          </div>

          <div className="space-y-2">
            <h4 className="font-medium">Key Influencing Factors</h4>
            <div className="grid gap-2">
              {sortedFeatures.map(([feature, importance]) => (
                <div
                  key={feature}
                  className="flex justify-between items-center"
                >
                  <span className="text-sm">{feature}</span>
                  <Progress value={importance * 100} className="w-32" />
                  <span className="text-sm w-16 text-right">
                    {(importance * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {Object.entries(appliedPatterns).length > 0 && (
            <div className="space-y-2">
              <h4 className="font-medium">Detected Patterns</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(appliedPatterns).map(([key, pattern]) => (
                  <Badge key={key} variant="secondary">
                    {pattern.type}: {pattern.value}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {meta_info && (
            <div className="text-sm text-muted-foreground">
              Generated at: {new Date(meta_info.generated_at).toLocaleString()}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
