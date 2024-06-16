import numpy as np
from sklearn.metrics import precision_recall_curve


class Common():

    def __init__(self):
        pass

    def threshold_calibration(self,model,X,y) -> tuple[list,float]:
        probabilities = model.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, probabilities)
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        # Find the threshold that maximizes F1 score
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        # Convert probabilities to binary predictions based on the threshold
        predictions = (probabilities > optimal_threshold).astype(int)
        return predictions, optimal_threshold