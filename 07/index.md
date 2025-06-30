# Recommendation Model

English | [简体中文](./index_zh-CN.md)

Platforms like YouTube, Netflix, and TikTok have gained popularity by using recommendation algorithms to suggest content that users might be interested in. Simply put, models learn the hidden mathematical relationships behind data. The Transformer architecture is mainly used for processing sequential data, while data annotation for recommendation content is typically matrix association data between users and content.

The feedback data annotation used in recommendations can be either explicit or implicit.

Explicit: Users directly annotate their preferences for content, such as ratings or reviews

| User        | Movie1   | Movie2   | Movie3   | Movie4   | Movie5   |
| ----------- | -------- | -------- | -------- | -------- | -------- |
| 123456789   | 3        | 3        | 3        | 3        | 3        |
| 325469787   | 5        | 3        | 3        | 3        | 3        |
| 258649112   | 7        | 3        | 3        | 3        | 3        |

Implicit: User preferences are expressed through watch time, clicks, purchase records, etc.

| User        | Content   | Watch Time |
| ----------- | --------- | -------- |
| 7337153711992174438 | 100074831 | 44652 |
| 5190801220865459604 | 100170790 | 121420 |
| 2293633612703952721 | 100510126 | 47744 |
| 5874973374932455844 | 100510126 | 32109 |
| 1173698801255170595 | 100676857 | 10512 |


Here we briefly introduce the matrix factorization model for computing implicit feedback, using the [Weighted Alternating Least Squares algorithm](http://yifanhu.net/PUB/cf.pdf) (WALS).

Weighted Alternating Least Squares is an optimization method for collaborative filtering, particularly suitable for handling implicit feedback data. Unlike explicit feedback, implicit feedback can only indicate user-content interactions but cannot directly reflect the degree of user preference. The WALS algorithm solves this problem through the following steps:

1. Convert the user-item interaction matrix to a confidence matrix, typically using the formula: confidence = 1 + α * observed value (where α is a hyperparameter)
2. Alternately optimize the user factor matrix and item factor matrix:
   - Fix item factors, solve for user factors
   - Fix user factors, solve for item factors
3. Each optimization is a weighted least squares problem, completed by solving linear equation systems

The advantage of this method is that it can effectively handle large-scale sparse data and improve recommendation quality by distinguishing between observed and unobserved interactions through a weighting mechanism. Here's a code example:

```python
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

class ImplicitMF:
    def __init__(self, n_factors=50, regularization=0.01, alpha=40, iterations=10):
        self.n_factors = n_factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
    
    def fit(self, user_item_matrix):
        # Convert observed interactions to confidence
        confidence = 1 + self.alpha * user_item_matrix.toarray()
        
        n_users, n_items = user_item_matrix.shape
        
        # Initialize user and item factor matrices
        self.user_factors = np.random.normal(size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(size=(n_items, self.n_factors))
        
        # Weighted Alternating Least Squares algorithm
        for _ in range(self.iterations):
            # Update user factors
            for u in range(n_users):
                # Create a diagonal matrix Cu where diagonal values are user u's confidence for all items
                # confidence[u] is user u's confidence vector for all items
                # flatten() converts multi-dimensional array to one-dimensional
                # np.diag() creates a diagonal matrix with values from the input vector on the diagonal
                Cu = np.diag(confidence[u].flatten())
                # This line calculates the coefficient matrix A in the ALS algorithm
                # self.item_factors.T is the transpose of the item factor matrix
                # Cu is the confidence diagonal matrix for user u on all items
                # self.item_factors is the item factor matrix
                # self.regularization * np.eye(self.n_factors) adds regularization term to prevent overfitting
                # The entire expression calculates the coefficient matrix for the linear equation system to solve user factors
                A = self.item_factors.T @ Cu @ self.item_factors + self.regularization * np.eye(self.n_factors)
                # This line calculates the right-hand side vector b in the ALS algorithm
                # self.item_factors.T is the transpose of the item factor matrix
                # Cu is the confidence diagonal matrix for user u on all items
                # (user_item_matrix[u].toarray().flatten() > 0) creates a boolean vector indicating which items user u interacted with
                # The entire expression calculates the right-hand side vector for the linear equation system to solve user factors
                b = self.item_factors.T @ Cu @ (user_item_matrix[u].toarray().flatten() > 0)
                # This line uses linear algebra to solve the equation system Ax=b, calculating user u's factor vector
                # np.linalg.solve function solves the linear equation system, where A is the coefficient matrix and b is the right-hand side vector
                # The solution is user u's latent factors, representing the user in the latent space
                # These factors will be multiplied with item factors to predict user preferences for items
                self.user_factors[u] = np.linalg.solve(A, b)
            
            # Update item factors
            for i in range(n_items):
                # Create a diagonal matrix Ci where diagonal values are all users' confidence for item i
                # confidence[:, i] is all users' confidence vector for item i
                # flatten() converts multi-dimensional array to one-dimensional
                # np.diag() creates a diagonal matrix with values from the input vector on the diagonal
                Ci = np.diag(confidence[:, i].flatten())
                # This line calculates the coefficient matrix A in the ALS algorithm
                # self.user_factors.T is the transpose of the user factor matrix
                # Ci is the confidence diagonal matrix for all users on item i
                # self.user_factors is the user factor matrix
                # self.regularization * np.eye(self.n_factors) adds regularization term to prevent overfitting
                # The entire expression calculates the coefficient matrix for the linear equation system to solve item factors
                A = self.user_factors.T @ Ci @ self.user_factors + self.regularization * np.eye(self.n_factors)
                # This line calculates the right-hand side vector b in the ALS algorithm
                # self.user_factors.T is the transpose of the user factor matrix
                # Ci is the confidence diagonal matrix for all users on item i
                # (user_item_matrix[:, i].toarray().flatten() > 0) creates a boolean vector indicating which users interacted with item i
                # The entire expression calculates the right-hand side vector for the linear equation system to solve item factors
                b = self.user_factors.T @ Ci @ (user_item_matrix[:, i].toarray().flatten() > 0)
                # This line uses linear algebra to solve the equation system Ax=b, calculating item i's factor vector
                # np.linalg.solve function solves the linear equation system, where A is the coefficient matrix and b is the right-hand side vector
                # The solution is item i's latent factors, representing the item in the latent space
                # These factors will be multiplied with user factors to predict user preferences for items
                self.item_factors[i] = np.linalg.solve(A, b)
    
    def recommend(self, user_id, n_items=10):
        # Calculate predicted ratings for user on all items
        # User factors multiplied by item factors represent similarity or compatibility in latent space
        # User factor vector represents user preferences in the latent feature space
        # Item factor vector represents item attributes in the same latent feature space
        # Their dot product (inner product) calculates their matching degree in the latent space
        # Higher dot product values indicate stronger user preference for that item, i.e., higher predicted rating
        scores = self.user_factors[user_id] @ self.item_factors.T
        # Return the top n items with highest scores
        return np.argsort(scores)[::-1][:n_items]

# Example usage
# Construct example data
users = [0, 0, 1, 1, 2, 2]
items = [0, 1, 0, 2, 1, 2]
watch_time = [120, 80, 95, 150, 200, 45]  # Watch time as implicit feedback

# Create user-item matrix
user_item_matrix = coo_matrix((watch_time, (users, items)), shape=(3, 3)).tocsr()

# Train model
model = ImplicitMF(n_factors=10, iterations=20)
model.fit(user_item_matrix)

# Recommend items for user 0
recommendations = model.recommend(user_id=0, n_items=2)
print(f"Recommended items for user 0: {recommendations}")
```
