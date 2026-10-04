import unittest
import pipeline

class PipelineTests(unittest.TestCase):
    def test_model_and_business_case(self):
        scored, _, auc = pipeline.train_and_score(pipeline.make_demo_accounts(n=800))
        case = pipeline.intervention_business_case(scored)
        self.assertGreater(auc, 0.7)
        self.assertAlmostEqual(case["net_value"], case["expected_revenue_saved"] - case["intervention_cost"])

if __name__ == "__main__":
    unittest.main()
