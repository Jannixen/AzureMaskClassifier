import unittest
from predictor import ImageMaskPredictor


class TestMaskCase(unittest.TestCase):
    def test_prediction(self):
        img = open('Test/Mask/2070.jpg', "rb")

        image_predictor = ImageMaskPredictor()
        prediction = image_predictor.classify_image(img)
        self.assertEqual(prediction, True)


class TestNonMaskCase(unittest.TestCase):
    def test_prediction(self):
        img = open('Test/Non Mask/real_01032.jpg', "rb")

        image_predictor = ImageMaskPredictor()
        prediction = image_predictor.classify_image(img)
        self.assertEqual(prediction, False)


if __name__ == '__main__':
    unittest.main()
