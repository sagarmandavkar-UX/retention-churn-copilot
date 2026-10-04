import unittest
import pipeline

class PipelineTests(unittest.TestCase):
    def test_model_and_business_case(self):
        scored, _, auc = pipeline.train_and_score(pipeline.make_demo_accounts(n=800))
        case = pipeline.intervention_business_case(scored)
        self.assertGreater(auc, 0.7)
        self.assertAlmostEqual(case["net_value"], case["expected_revenue_saved"] - case["intervention_cost"])

    def test_holdout_lift_and_threshold_curve(self):
        scored, _, _ = pipeline.train_and_score(pipeline.make_demo_accounts(n=1200))
        lift = pipeline.lift_table(scored)
        curve = pipeline.threshold_curve(scored)
        self.assertGreater(lift.iloc[0]["lift"], 1)
        self.assertTrue(curve["precision"].between(0, 1).all())
        self.assertTrue(curve["recall"].between(0, 1).all())

if __name__ == "__main__":
    unittest.main()
