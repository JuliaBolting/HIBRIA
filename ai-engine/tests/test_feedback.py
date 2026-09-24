import unittest

from pipeline.persistence.feedback_repository import FeedbackRepository


class FeedbackRepositoryTests(unittest.TestCase):
    def test_ratings_one_and_two_are_negative(self):
        self.assertEqual(FeedbackRepository.category(1), "negativa")
        self.assertEqual(FeedbackRepository.category(2), "negativa")

    def test_rating_three_is_neutral(self):
        self.assertEqual(FeedbackRepository.category(3), "neutra")

    def test_ratings_four_and_five_are_positive(self):
        self.assertEqual(FeedbackRepository.category(4), "positiva")
        self.assertEqual(FeedbackRepository.category(5), "positiva")


if __name__ == "__main__":
    unittest.main()
