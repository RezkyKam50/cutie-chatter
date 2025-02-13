import numpy as np

class DynamicSequentialReflection:
    def __init__(self, weights, bias, max_interactions=50, max_duration=10):
        """
        Initialize the AttachmentCalculator.
        
        Parameters:
        - weights (list of float): Weights [w1, w2, w3, w4]
        - bias (float): Bias term
        - max_interactions (int): Maximum expected interactions (for normalization)
        - max_duration (int): Maximum expected duration (for normalization)
        """
        self.weights = weights
        self.bias = bias
        self.max_interactions = max_interactions
        self.max_duration = max_duration

    @staticmethod
    def sigmoid(x):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def SelfInterest(self, interactions, sentiment_consistency, 
                             avg_cosine_similarity, engagement_duration):
        """
        Calculate the attachment probability.
        
        Parameters:
        - interactions (float): Number of interactions (I)
        - sentiment_consistency (float): Sentiment consistency (S_c) [0 to 1]
        - avg_cosine_similarity (float): Average cosine similarity (CS̄) [0 to 1]
        - engagement_duration (float): Engagement duration (D)
        
        Returns:
        - float: Attachment probability ATTACHMENT
        """
        # Normalize interaction count and engagement duration
        norm_interactions = interactions / self.max_interactions
        norm_engagement_duration = engagement_duration / self.max_duration

        # Weighted sum of inputs
        z = (self.weights[0] * norm_interactions +
             self.weights[1] * sentiment_consistency +
             self.weights[2] * avg_cosine_similarity +
             self.weights[3] * norm_engagement_duration +
             self.bias)

        # Apply sigmoid function
        return self.sigmoid(z)

    def SelfRegulation(self, attachment):
        """
        Calculate the dynamic threshold DTS based on ATTACHMENT.
        
        Parameters:
        - attachment (float): Attachment probability
        
        Returns:
        - float: Dynamic threshold (DTS)
        """
        return 1 - attachment

# Example usage
if __name__ == "__main__":
    # Example weights and bias
    weights = [0.3, 0.5, 0.1, 0.1]  # [interactions, sentiment_consistency, avg_cosine_similarity, engagement_duration]
    bias = 0.0

    # Create an instance of AttachmentCalculator
    calculator = DynamicSequentialReflection(weights, bias)

    # Example input values
    interactions = 35
    sentiment_consistency = 1.0
    avg_cosine_similarity = 1.0
    engagement_duration = 1.0

    # Compute ATTACHMENT and DTS
    attachment = calculator.SelfInterest(
        interactions, 
        sentiment_consistency, 
        avg_cosine_similarity, 
        engagement_duration
    )
    dynamic_threshold = calculator.SelfRegulation(attachment)

    print(f"Attachment Probability (ATTACHMENT): {attachment:.4f}")
    print(f"Dynamic Threshold (DTS): {dynamic_threshold:.4f}")