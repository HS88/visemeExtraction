import json
import numpy as np


class Viseme(object):
    def __init__(self, score: float):
        self.code = score
        self.duration = 1
        pass

    def __eq__(self, other):
        return self.code == other.code


class VisemeClassifier(object):

    def __init__(self, viseme_file: str):
        # Load viseme data from JSON file
        with open(viseme_file, 'r') as json_file:
            self.viseme_data = json.load(json_file)
        self.visemes = self.viseme_data['viseme_lut']

    def match_viseme(self, score: float) -> str:
        viseme_scores = {}
        for (key, value) in self.visemes:
            viseme_scores[key] = abs(score - value)
        return min(viseme_scores, key=viseme_scores.get)

    def parse_features(self, features: np.ndarray):
        visemes = []
        for feature in features:
            height = np.linalg.norm(feature[0] - feature[1])
            width = np.linalg.norm(feature[2] - feature[3])
            score = height / width
            visemes.append(self.match_viseme(score))

    # def __init__(self, features: np.ndarray):
    #     self.visemes = []
    #     self.scores = []
    #     for feature in features:
    #         height = np.linalg.norm(feature[0] - feature[1])
    #         width = np.linalg.norm(feature[2] - feature[3])
    #         score = height / width
    #         self.scores.append(score)
    #         viseme = Viseme(score)
    #         if not self.visemes or self.visemes[-1] != viseme:
    #             self.visemes.append(viseme)
    #         else:
    #             self.visemes[-1].duration += 1
