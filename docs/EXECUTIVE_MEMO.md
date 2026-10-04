# Executive memo

## Demonstration finding

The holdout model provides useful rank ordering of churn risk. The highest-risk decile concentrates materially more churn than the population baseline, creating a practical review queue.

## Decision use

Choose the intervention threshold using capacity, precision/recall, customer experience, and expected economics together. A lower threshold increases coverage but consumes more retention capacity and sends more offers to accounts that would not churn.

## Recommendation

Use the model for prioritization, not autonomous action. Run a randomized retention experiment to measure the true save rate and offer cost, then replace the scenario economics with incremental outcomes.
