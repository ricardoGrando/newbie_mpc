import gym
import cvxpy
import numpy as np

# Define the CartPole environment
env = gym.make('CartPole-v1')

# MPC parameters
horizon = 10  # MPC prediction horizon
num_actions = env.action_space.n

# Define the MPC optimization problem
state = cvxpy.Variable((4, horizon + 1))  # State trajectory
action = cvxpy.Variable((num_actions, horizon))  # Action trajectory

# Dynamics constraints
A, B = np.eye(4), np.zeros((4, num_actions))
constraints = [state[:, 1:] == A @ state[:, :-1] + B @ action]

# Bounds on actions and state
action_bounds = 1.0
state_bounds = 10.0
constraints += [cvxpy.norm(action, 'inf') <= action_bounds]
constraints += [cvxpy.norm(state, 'inf') <= state_bounds]

# Modified objective function (minimize deviations from the upright position)
penalty_weight = 0.1
objective = cvxpy.Minimize(cvxpy.sum_squares(action) + penalty_weight * cvxpy.sum_squares(state[2, 1:]))

# Define the MPC problem
problem = cvxpy.Problem(objective, constraints)

# Simulation
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(env.spec.max_episode_steps):
        # Solve MPC problem
        problem.solve()

        # Take the first action from the optimal trajectory
        action_optimal = action.value[:, 0]
        next_state, reward, done, _ = env.step(np.argmax(action_optimal))

        # Update the state for the next iteration
        state = next_state
        total_reward += reward

        # Render the environment (optional)
        # env.render()

        if done:
            break

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Close the environment
env.close()
