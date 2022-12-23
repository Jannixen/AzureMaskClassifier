from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

from config import PREDICTION_KEY_CUSTOM_VISION, PROJECT_ID_MASKS

ENDOPINT_CUSTOM_VISION = "https://machinelearning1-prediction.cognitiveservices.azure.com/"

ITERATION_MASKS = "Iteration1"



class ImageMaskPredictor:

    def __init__(self):
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY_CUSTOM_VISION})
        self.predictor = CustomVisionPredictionClient(ENDOPINT_CUSTOM_VISION, prediction_credentials)

    def classify_image(self, image):
        probabilities = self.detect_mask(image)
        if probabilities['mask'] > probabilities['non-mask']:
            return True
        return False

    def detect_mask(self, image):
        results = self.predictor.classify_image(PROJECT_ID_MASKS, ITERATION_MASKS, image.read())
        return self.make_results_dict(results)

    @staticmethod
    def make_results_dict(results):
        results_dict = {}
        for prediction in results.predictions:
            results_dict[prediction.tag_name] = prediction.probability
        return results_dict
