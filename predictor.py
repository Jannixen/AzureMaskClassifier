import environ
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

env = environ.Env()
environ.Env.read_env()

PREDICTION_KEY_CUSTOM_VISION = env("PREDICTION_KEY_CUSTOM_VISION")
ENDPOINT_CUSTOM_VISION = env("ENDOPINT_CUSTOM_VISION ")

ITERATION_MASKS = env("ITERATION")
PROJECT_ID_MASKS = env("PROJECT_ID")


class ImageMaskPredictor:

    def __init__(self):
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY_CUSTOM_VISION})
        self.predictor = CustomVisionPredictionClient(ENDPOINT_CUSTOM_VISION, prediction_credentials)

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
