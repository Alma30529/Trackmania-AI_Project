import gymnasium as gym
import numpy as np
import random
import time
import json
import zipfile
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from env import TrackmaniaPixelEnv

# 1. PPO LOGGING CALLBACK
# 
class PPOBenchmarkCallback(BaseCallback):
    """Custom callback to track the all-time best reward during PPO training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.timesteps = []
        self.best_rewards = []
        self.current_best = -float('inf')
        self.ep_reward = 0

    def _on_step(self) -> bool:
        # Accumulate reward for the current episode
        self.ep_reward += self.locals["rewards"][0]
        
        # If episode is done (terminated or truncated)
        if self.locals["dones"][0]:
            if self.ep_reward > self.current_best:
                self.current_best = self.ep_reward
            
            # Log the data point
            self.timesteps.append(self.num_timesteps)
            self.best_rewards.append(float(self.current_best))
            self.ep_reward = 0 
            
        return True

# ==========================================
# 2. EA LOGIC (TIMESTEP CONSTRAINED)
# ==========================================
def evaluate_ea(env, sequence):
    obs, _ = env.reset()
    total_reward = 0
    steps_taken = 0
    
    for action in sequence:
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps_taken += 1
        if terminated or truncated:
            break
            
    return total_reward, steps_taken

def mutate(sequence, action_space_size, mutation_rate=0.2):
    return [random.randint(0, action_space_size - 1) if random.random() < mutation_rate else act for act in sequence]

def run_ea_benchmark(env, timestep_budget, seq_len=40, pop_size=5):
    print(f"\n STARTING EA TRIAL (Budget: {timestep_budget} steps)")
    
    population = [[random.randint(0, 2) for _ in range(seq_len)] for _ in range(pop_size)]
    ea_global_best = -float('inf')
    
    timesteps_log = []
    best_rewards_log = []
    steps_used = 0
    gen = 1
    
    while steps_used < timestep_budget:
        scores = []
        for i, seq in enumerate(population):
            if steps_used >= timestep_budget:
                break # Hard cutoff
                
            score, steps_taken = evaluate_ea(env, seq)
            steps_used += steps_taken
            scores.append(score)
            
            # Track global best
            if score > ea_global_best:
                ea_global_best = score
                
            # Log progress
            timesteps_log.append(steps_used)
            best_rewards_log.append(float(ea_global_best))
            
        print(f"EA Gen {gen} | Steps Used: {steps_used}/{timestep_budget} | Best Score: {ea_global_best:.2f}")
        
        if scores:
            best_idx = np.argmax(scores)
            best_seq = population[best_idx]
            population = [mutate(best_seq, 3) for _ in range(pop_size)]
            population[0] = best_seq # Elitism
        gen += 1
        
    return timesteps_log, best_rewards_log

# ==========================================
# 3. MAIN BENCHMARK RUNNER
# ==========================================
def main():
    # SET BUDGET (Keep it low for testing, e.g., 2000. Increase to 10000+ for real results)
    TIMESTEP_BUDGET = 2000 
    env = TrackmaniaPixelEnv()
    
    print("="*50)
    print(f" TRACKMANIA AI SHOWDOWN: PPO vs EA")
    print(f" TIMESTEP BUDGET: {TIMESTEP_BUDGET}")
    print("="*50)
    time.sleep(2)

    # --- RUN PPO ---
    print("\ STARTING PPO TRIAL (CNN Policy)... Please click into Trackmania")
    ppo_callback = PPOBenchmarkCallback()
    model = PPO("CnnPolicy", env, verbose=0, n_steps=512, batch_size=64)
    model.learn(total_timesteps=TIMESTEP_BUDGET, callback=ppo_callback)
    
    ppo_timesteps = ppo_callback.timesteps
    ppo_rewards = ppo_callback.best_rewards

    # --- RUN EA ---
    ea_timesteps, ea_rewards = run_ea_benchmark(env, TIMESTEP_BUDGET)

    # --- SAVE & ZIP ---
    data = {
        "budget": TIMESTEP_BUDGET,
        "ppo": {"timesteps": ppo_timesteps, "best_rewards": ppo_rewards},
        "ea": {"timesteps": ea_timesteps, "best_rewards": ea_rewards}
    }
    
    json_name = "ppo_vs_ea_data.json"
    zip_name = "ppo_vs_ea_results.zip"
    
    with open(json_name, "w") as f:
        json.dump(data, f, indent=4)
        
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_name)
    os.remove(json_name)
    print(f"\n Data saved and compressed to '{zip_name}'")

    # --- PLOT ---
    plt.figure(figsize=(10, 6))
    plt.plot(ppo_timesteps, ppo_rewards, label='PPO (CNN Deep Learning)', color='purple', linewidth=2)
    plt.plot(ea_timesteps, ea_rewards, label='EA (Heuristic Search)', color='blue', linewidth=2)
    
    plt.title(f'PPO vs EA: Learning Curve ({TIMESTEP_BUDGET} Timesteps)')
    plt.xlabel('Environment Timesteps (Interactions)')
    plt.ylabel('All-Time Best Reward Found')
    plt.legend()
    plt.grid(True)
    plt.savefig('ppo_vs_ea_plot.png')
    print(" Graph saved as 'ppo_vs_ea_plot.png'")
    plt.show()

if __name__ == "__main__":
    main()