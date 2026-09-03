import numpy as np
import random

# Define the objective function
def objective_function(params):
    x, y, z = params[0], params[1], params[2]
    return (x-4)**2 + (y-5)**2 + (z+6)**2

# Define the bounds of the search space
bounds = np.array([[-10, -10, -10], [10, 10, 10]])

# Define the parameters of the optimization that control the movement of the
# particles in the search space.
n_particles = 10
max_iterations = 30

w = 0.5 #the inertia weight that balances the particle's current velocity
#high value of w leads to more exploration and less exploitation

#c1 and c2 are the acceleration coefficients that control the influence of the
# particle's best personal position and the global best position on its movement.
c1 = 0.8  #Cognitive component - represents the particle's tendency to move towards its best personal position
c2 = 0.9  #Social component, which represents the particle's tendency to move towards the global best position found by the swarm

# Initialize the particles and velocities
particles = np.random.uniform(low=bounds[0], high=bounds[1], size=(n_particles, 3))
velocities = np.zeros((n_particles, 3))

# Initialize the best positions and best costs
best_positions = particles.copy()
best_costs = np.array([objective_function(p) for p in particles])

# Initialize the global best position and global best cost
global_best_position = particles[0].copy()
global_best_cost = best_costs[0]

# Perform the optimization
for i in range(max_iterations):
    # Update the velocities
    r1 = np.random.rand(n_particles, 3) #Random matrix used to compute the cognitive component of the veocity update
    r2 = np.random.rand(n_particles, 3) #Random matrix used to compute the social component of the veocity update


    #Cognitive component is calculated by taking the difference between the
    #particle's current position and its best personal position found so far,
    #and then multiplying it by a random matrix r1 and a cognitive acceleration coefficient c1.
    cognitive_component = c1 * r1 * (best_positions - particles)

    #The social component represents the particle's tendency to move towards the
    #global best position found by the swarm. It is calculated by taking the
    #difference between the particle's current position and the global best position
    # found by the swarm, and then multiplying it by a random matrix r2 and a
    #social acceleration coefficient c2.
    social_component = c2 * r2 * (global_best_position - particles)

    #The new velocity of the particle is computed by adding the current velocity
    #to the sum of the cognitive and social components, multiplied by the inertia
    #weight w. The new velocity is then used to update the position of the
    #particle in the search space.
    velocities = w * velocities + cognitive_component + social_component

    # Update the particles
    particles += velocities

    # Enforce the bounds of the search space
    particles = np.clip(particles, bounds[0], bounds[1])

    # Evaluate the objective function
    costs = np.array([objective_function(p) for p in particles])

    # Update the best positions and best costs
    is_best = costs < best_costs
    best_positions[is_best] = particles[is_best]
    best_costs[is_best] = costs[is_best]

    # Update the global best position and global best cost
    global_best_index = np.argmin(best_costs)
    global_best_position = best_positions[global_best_index].copy()
    global_best_cost = best_costs[global_best_index]

    # Print the progress
    print(f'Iteration {i+1}: Best Cost = {global_best_cost:.6f}')

# Print the results
print('Global Best Position:', global_best_position)
print('Global Best Cost:', global_best_cost)