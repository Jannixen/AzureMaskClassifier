import os
import time
import unittest

from predictor import ImageMaskPredictor


class TestMaskCase(unittest.TestCase):
    def test_prediction(self):
        img = open('TestFiles/Mask/2070.jpg', "rb")

        image_predictor = ImageMaskPredictor()
        prediction = image_predictor.classify_image(img)
        self.assertEqual(prediction, True)


class TestNonMaskCase(unittest.TestCase):
    def test_prediction(self):
        img = open('TestFiles/Non Mask/real_01032.jpg', "rb")

        image_predictor = ImageMaskPredictor()
        prediction = image_predictor.classify_image(img)
        self.assertEqual(prediction, False)


class TestNonMaskCaseAll(unittest.TestCase):
    def test_prediction(self):
        image_predictor = ImageMaskPredictor()
        errors = 0

        for file in os.scandir("TestFiles/Non Mask/"):
            img = open(file, "rb")
            prediction = image_predictor.classify_image(img)
            if prediction:
                errors += 1
            time.sleep(1)
        print("Błąd klasyfikacji dla danych testowych - przypadki non-mask", errors / 50)


class TestMaskCaseAll(unittest.TestCase):
    def test_prediction(self):
        image_predictor = ImageMaskPredictor()
        errors = 0

        for file in os.scandir("TestFiles/Mask/"):
            img = open(file, "rb")
            prediction = image_predictor.classify_image(img)
            if not prediction:
                errors += 1
            time.sleep(1)
        print("Błąd klasyfikacji dla danych testowych - przypadki mask", errors / 50)


if __name__ == '__main__':
    unittest.main()
