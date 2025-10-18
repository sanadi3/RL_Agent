from agent import QLearningAgent    # tabular learning
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import streamlit as st
import time
from dqn import DQNAgent    # deep learning agent


# Streamlit dashboard
st.title("Q-learning vs Deep Q-learning on FrozenLake")

if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'algorithm' not in st.session_state:
    st.session_state.algorithm = None

# Streamlit sidebar
num_episodes = st.sidebar.slider("Training Episodes", 100, 50000, 5000, step=100)
max_steps = st.sidebar.slider("Max steps per episode", 10, 500, 100, step=10)

epsilon = st.sidebar.slider("Initial exploration rate", 0.01, 1.0, 1.0)
epsilon_min = st.sidebar.slider("Minimum epsilon", 0.001, 0.1, 0.05)
epsilon_decay = st.sidebar.slider("Epsilon decay", 0.9, 0.9999, 0.995)
alpha = st.sidebar.slider("Learning rate (alpha)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount (gamma)", 0.1, 0.99, 0.95)

# DQN controls
replay_batch_size = st.sidebar.slider("DQN replay batch size", 16, 256, 64, step=16)
dqn_start_training = st.sidebar.slider("DQN warmup (min experiences before replay)", 0, 5000, 100, step=10)

# Create FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Choose algorithm
algorithm = st.sidebar.selectbox("Algorithm", ["Q-Learning (tabular)", "Deep Q-Learning (neural network)"])

# Create new agent if algorithm changed
if st.session_state.algorithm != algorithm:
    st.session_state.trained = False
    st.session_state.algorithm = algorithm
    
    if algorithm == "Q-Learning (tabular)":
        st.session_state.agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon, 
                                                 epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
        st.sidebar.write("Using tabular Q-Learning")
    else: 
        state_size = env.observation_space.n
        action_size = env.action_space.n
        st.session_state.agent = DQNAgent(state_size, action_size, alpha=alpha, gamma=gamma, 
                                          epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)
        st.sidebar.write("Using PyTorch DQN. DQN needs more episodes + replay buffer to work well")

agent = st.session_state.agent

train_button = st.button("Train Agent")
watch_button = st.button("Watch Trained Agent")
status_text = st.empty()
progress_bar = st.empty()
chart_placeholder = st.empty()

rewards_per_episode = []

# Encode discrete state for DQN
def one_hot_state(s, state_size):
    """Convert discrete state to one-hot encoded vector"""
    vec = np.zeros(state_size, dtype=np.float32)
    vec[s] = 1.0
    return vec

if train_button:
    status_text.text("Training started...")
    progress_bar = st.progress(0)
    start_time = time.time()

    # Training loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            if algorithm == "Q-Learning (tabular)":
                action = agent.choose_action(obs)
            else:
                s_vec = one_hot_state(obs, env.observation_space.n)
                action = agent.choose_action(s_vec)
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if algorithm == "Q-Learning (tabular)":
                shaped_reward = reward
            else:
                if terminated and reward > 0:
                    shaped_reward = 1.0
                elif terminated and reward == 0:
                    shaped_reward = -1.0
                else:
                    shaped_reward = -0.01

            if algorithm == "Q-Learning (tabular)":
                # Update Q-values
                agent.update(obs, action, reward, next_obs)
            else:
                # Store experience and trigger replay when enough experiences
                next_vec = one_hot_state(next_obs, env.observation_space.n)
                agent.remember((one_hot_state(obs, env.observation_space.n), action, shaped_reward, next_vec, float(done)))
                if len(agent.memory) >= max(dqn_start_training, replay_batch_size):
                    agent.replay(batch_size=replay_batch_size)
            
            obs = next_obs
            total_reward += reward
            
            if done:
                break
        
        # Decay epsilon for next episode
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        rewards_per_episode.append(total_reward)

        # Update UI every 100 episodes
        if (episode + 1) % 100 == 0 or (episode + 1) == num_episodes:
            percent_complete = int((episode + 1) * 100 / num_episodes)
            progress_bar.progress(percent_complete)
            status_text.text(f"Episode {episode+1}/{num_episodes} | Last reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
            
            window = 100
            if len(rewards_per_episode) >= 2:
                smooth = np.convolve(rewards_per_episode, np.ones(min(window, len(rewards_per_episode)))/min(window, len(rewards_per_episode)), mode='valid')
                chart_placeholder.line_chart(smooth)
        
    st.session_state.trained = True

    elapsed = time.time() - start_time
    status_text.text(f"Training completed in {elapsed:.1f}s. Final epsilon = {agent.epsilon:.3f}")

    st.success("Training complete!")

    if len(rewards_per_episode) > 0:
        fig, ax = plt.subplots()
        window = max(1, min(100, len(rewards_per_episode)))
        if len(rewards_per_episode) >= window and window > 1:
            smoothed = np.convolve(rewards_per_episode, np.ones(window) / window, mode='valid')
            ax.plot(smoothed)
        else:
            ax.plot(rewards_per_episode)
        ax.set_title("Rewards over time")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        st.pyplot(fig)

if watch_button:
    if not st.session_state.trained:
        st.error("⚠️ Please train the agent first by clicking 'Train Agent'!")
    else:
        st.info(f"🎬 Watching trained {st.session_state.algorithm} agent...")
        
        # Store original epsilon and set to 0 for pure exploitation
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        # FIX: slippery false
        watch_env = gym.make('FrozenLake-v1', is_slippery=False)
        obs, _ = watch_env.reset()
        total_reward = 0
        grid_size = int(np.sqrt(watch_env.observation_space.n))
        
        # Get the map for visualization
        desc = watch_env.unwrapped.desc.astype(str)
        
        placeholder = st.empty()
        steps_taken = 0

        for step in range(50):
            # Render grid with emojis
            grid_display = []
            for i in range(grid_size):
                row = []
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx == obs:
                        row.append("🤖")  # Agent
                    elif desc[i][j] == 'S':
                        row.append("🟢")  # Start
                    elif desc[i][j] == 'G':
                        row.append("🎯")  # Goal
                    elif desc[i][j] == 'H':
                        row.append("🕳️")  # Hole
                    else:
                        row.append("❄️")  # Frozen
                grid_display.append("  ".join(row))
            
            # Choose action based on algorithm
            if st.session_state.algorithm == "Q-Learning (tabular)":
                # Use greedy policy (no exploration)
                q_values = agent.Q[obs]
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                action = np.random.choice(best_actions)
            else:
                # DQN: use greedy policy
                s_vec = one_hot_state(obs, watch_env.observation_space.n)
                action = agent.choose_action(s_vec, training=False)
            
            # Add debug info for Q-learning
            debug_info = ""
            if st.session_state.algorithm == "Q-Learning (tabular)":
                debug_info = f"\nQ-values: {[f'{q:.2f}' for q in q_values]}"
                debug_info += f"\nAction taken: {action} (0=Left, 1=Down, 2=Right, 3=Up)"
            
            display_text = "\n".join(grid_display)
            display_text += f"\n\nStep: {steps_taken} | Total Reward: {total_reward:.2f}"
            display_text += debug_info
            placeholder.text(display_text)
            time.sleep(0.5)
            
            next_obs, reward, terminated, truncated, _ = watch_env.step(action)
            obs = next_obs
            total_reward += reward
            steps_taken += 1
            
            if terminated or truncated:
                # Show final state
                grid_display = []
                for i in range(grid_size):
                    row = []
                    for j in range(grid_size):
                        idx = i * grid_size + j
                        if idx == obs:
                            if reward > 0:
                                row.append("🏆")  # Won!
                            else:
                                row.append("💀")  # Fell in hole
                        elif desc[i][j] == 'S':
                            row.append("🟢")
                        elif desc[i][j] == 'G':
                            row.append("🎯")
                        elif desc[i][j] == 'H':
                            row.append("🕳️")
                        else:
                            row.append("❄️")
                    grid_display.append("  ".join(row))
                
                display_text = "\n".join(grid_display)
                display_text += f"\n\nStep: {steps_taken} | Total Reward: {total_reward:.2f}"
                if reward > 0:
                    display_text += "\n\n✅ SUCCESS! Reached the goal!"
                else:
                    display_text += "\n\n❌ FAILED! Fell into a hole or ran out of time."
                placeholder.text(display_text)
                break
        
        watch_env.close()
        
        # Restore original epsilon
        agent.epsilon = original_epsilon
        
        if total_reward > 0:
            st.success(f"🎉 Agent succeeded in {steps_taken} steps!")
        else:
            st.warning(f"Agent failed after {steps_taken} steps. Try training more!")
        