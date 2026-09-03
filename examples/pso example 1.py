import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def train_network_forward_propagation(weights, hidden_layer_size, X_train_data, y_train_data):
    hidden_layer_weights = weights[:4*hidden_layer_size].reshape(4, hidden_layer_size) 
    output_layer_weights = weights[4*hidden_layer_size:].reshape(hidden_layer_size, 3) 
    
    # Forward pass
    hidden_layer_input = np.dot(X_train_data, hidden_layer_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_layer_weights)
    output_layer_output = sigmoid(output_layer_input)
    
    # Loss calculation (Mean Squared Error)
    k = y_train_data - output_layer_output
    loss = np.mean(np.square(k))
    return loss

def evaluate_network(weights, hidden_layer_size, X_test, y_test):
    hidden_layer_weights = weights[:4*hidden_layer_size].reshape(4, hidden_layer_size) 
    output_layer_weights = weights[4*hidden_layer_size:].reshape(hidden_layer_size, 3)
    
    # Forward pass
    hidden_layer_input = np.dot(X_test, hidden_layer_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_layer_weights)
    output_layer_output = sigmoid(output_layer_input)
    
    # Convert logits to probabilities
    predictions = softmax(output_layer_output)
    
    # Choose the class with the highest probability
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    true_classes = np.asarray(true_classes)

    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    
    # Generate a confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    
    return accuracy, conf_matrix

def pso_iris(num_particles, num_iterations, hidden_layer_size, X_train_data, y_train_data):
    # PSO parameters
    num_dimensions = 4 * hidden_layer_size + hidden_layer_size * 3
    positions = np.random.rand(num_particles, num_dimensions) - 0.5  # Initialize positions
    velocities = np.zeros_like(positions)  # Initialize velocities
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([train_network_forward_propagation(p, hidden_layer_size, X_train_data, y_train_data) for p in positions])
    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)

    # PSO loop
    w = 0.5  # inertia
    c1 = 2  # cognitive parameter
    c2 = 2  # social parameter
    for i in range(num_iterations):
        for j in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[j] = w * velocities[j] + c1 * r1 * (pbest_positions[j] - positions[j]) + c2 * r2 * (gbest_position - positions[j])
            positions[j] += velocities[j]
            
            current_score = train_network_forward_propagation(positions[j], hidden_layer_size, X_train_data, y_train_data)
            if current_score < pbest_scores[j]:
                pbest_scores[j] = current_score
                pbest_positions[j] = positions[j]
        
        # Update global best
        current_gbest_score = np.min([train_network_forward_propagation(p, hidden_layer_size, X_train_data, y_train_data) for p in positions])
        if current_gbest_score < gbest_score:
            gbest_score = current_gbest_score
            gbest_position = positions[np.argmin([train_network_forward_propagation(p, hidden_layer_size, X_train_data, y_train_data) for p in positions])]

        print(f"Iteration {i+1} - Best Loss: {gbest_score}")

    print("Optimal Weights Found by PSO:", gbest_position)
    return gbest_position

# Load and prepare the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Assuming best_position contains the optimal weights found by PSO
hidden_layer_size = 6
iris_classifier = pso_iris(30, 50, hidden_layer_size, X_test_scaled, y_test)
accuracy, conf_matrix = evaluate_network(iris_classifier, hidden_layer_size, X_test_scaled, y_test)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)