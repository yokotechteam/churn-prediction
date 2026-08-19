import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import DEFAULT_CUSTOMER, load_model, normalise_customer, score_customer


ROOT = Path(__file__).resolve().parents[1]


class PredictionLogicTests(unittest.TestCase):
    def test_default_customer_is_valid_and_scores_to_a_probability(self):
        customer, errors = normalise_customer(DEFAULT_CUSTOMER)

        self.assertEqual(errors, {})
        result = score_customer(customer)
        self.assertIsInstance(result["probability"], float)
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)
        self.assertEqual(result["percentage"], round(result["probability"] * 100, 1))
        self.assertIsInstance(result["churn"], bool)

    def test_model_is_cached_within_the_process(self):
        self.assertIs(load_model(), load_model())

    def test_invalid_values_return_specific_field_errors(self):
        customer, errors = normalise_customer(
            {**DEFAULT_CUSTOMER, "contract": "lifetime", "tenure": -1, "monthlycharges": "nan"}
        )

        self.assertEqual(customer["tenure"], None)
        self.assertIn("contract", errors)
        self.assertIn("tenure", errors)
        self.assertIn("monthlycharges", errors)

    def test_inconsistent_service_selections_are_rejected(self):
        _, errors = normalise_customer(
            {
                **DEFAULT_CUSTOMER,
                "phoneservice": "no",
                "multiplelines": "yes",
                "internetservice": "no",
                "onlinesecurity": "yes",
            }
        )

        self.assertIn("multiplelines", errors)
        self.assertIn("onlinesecurity", errors)


class StreamlitInterfaceTests(unittest.TestCase):
    def test_streamlit_app_renders_and_scores_the_default_form(self):
        app_test = AppTest.from_file(str(ROOT / "streamlit_app.py"))

        app_test.run(timeout=30)
        self.assertEqual(len(app_test.exception), 0)
        self.assertEqual(app_test.title[0].value, "Churn Lens")
        self.assertEqual(app_test.button[0].label, "Calculate churn risk")

        app_test.button[0].click().run(timeout=30)
        self.assertEqual(len(app_test.exception), 0)
        self.assertEqual(app_test.metric[0].label, "Estimated likelihood of churn")
        self.assertTrue(app_test.metric[0].value.endswith("%"))


if __name__ == "__main__":
    unittest.main()
