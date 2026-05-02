import numpy as np
import random
import time
import json
import zipfile
import os
import matplotlib.pyplot as plt
from env import TrackmaniaPixelEnv

# --- EA LOGIC ---
def evaluate_ea(env, sequence):
    obs, _ = env.reset()
    total_reward = 0
    for action in sequence:
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def mutate(sequence, action_space_size, mutation_rate=0.2):
    return [random.randint(0, action_space_size - 1) if random.random() < mutation_rate else act for act in sequence]

# --- PSO LOGIC ---
class Particle:
    def __init__(self, seq_len, action_size):
        self.position = np.random.uniform(0, action_size - 1, size=seq_len)
        self.velocity = np.random.uniform(-1, 1, size=seq_len)
        self.best_pos = np.copy(self.position)
        self.best_score = -float('inf')

def run_benchmarks():
    env = TrackmaniaPixelEnv()
    seq_len = 40  
    gens = 5       # Reduced to 5 iterations
    pop_size = 5   # Reduced population to minimize waiting/resets
    
    ea_history = []
    pso_history = []

    # 1. RUN EA
    print("\n" + "="*40)
    print("🚀 STARTING EA TRIALS (5 Generations)")
    print("="*40)
    population = [[random.randint(0, 2) for _ in range(seq_len)] for _ in range(pop_size)]
    
    ea_global_best = -float('inf') # Add this tracker
    
    for g in range(gens):
        scores = [evaluate_ea(env, seq) for seq in population]
        best_idx = np.argmax(scores)
        
        # Update global best if the current generation beats it
        if scores[best_idx] > ea_global_best:
            ea_global_best = scores[best_idx]
            
        ea_history.append(float(ea_global_best)) # Log the global best, not current gen
        print(f"✅ EA Gen {g+1} Complete: Best Score = {ea_global_best:.2f}")

    # 2. RUN PSO
    print("\n" + "="*40)
    print("🚀 STARTING PSO TRIALS (5 Generations)")
    print("="*40)
    particles = [Particle(seq_len, 3) for _ in range(pop_size)]
    global_best_pos = np.random.uniform(0, 2, size=seq_len)
    global_best_score = -float('inf')

    for g in range(gens):
        for p in particles:
            discrete_seq = np.round(np.clip(p.position, 0, 2)).astype(int)
            score = evaluate_ea(env, discrete_seq)
            
            if score > p.best_score:
                p.best_score = score
                p.best_pos = np.copy(p.position)
            if score > global_best_score:
                global_best_score = score
                global_best_pos = np.copy(p.position)
        
        pso_history.append(float(global_best_score))
        print(f"✅ PSO Gen {g+1} Complete: Best Score = {global_best_score:.2f}")

        for p in particles:
            r1, r2 = np.random.rand(seq_len), np.random.rand(seq_len)
            p.velocity = 0.7 * p.velocity + 2.0 * r1 * (p.best_pos - p.position) + 2.0 * r2 * (global_best_pos - p.position)
            p.position = np.clip(p.position + p.velocity, 0, 2)

    return ea_history, pso_history

def save_and_zip_results(ea_res, pso_res):
    data = {
        "ea_scores": ea_res,
        "pso_scores": pso_res,
        "generations": len(ea_res)
    }
    
    # Write to JSON
    json_filename = "benchmark_data.json"
    with open(json_filename, "w") as f:
        json.dump(data, f, indent=4)
        
    # Compress to ZIP
    zip_filename = "trackmania_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_filename)
        
    # Cleanup raw json to keep folder clean
    os.remove(json_filename)
    print(f"\n📁 Data successfully saved and compressed into '{zip_filename}'")

if __name__ == "__main__":
    ea_res, pso_res = run_benchmarks()
    
    # Save the data
    save_and_zip_results(ea_res, pso_res)
    
    # Generate the immediate plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 6), ea_res, label='EA (Discrete Mutation)', marker='^', color='blue', linewidth=2)
    plt.plot(range(1, 6), pso_res, label='PSO (Continuous Swarm)', marker='o', color='red', linewidth=2)
    plt.title('EA vs PSO: Fitness Progression (5 Generations)')
    plt.xlabel('Generation')
    plt.ylabel('Highest Reward Found')
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True)
    
    plot_file = 'ea_vs_pso_plot.png'
    plt.savefig(plot_file)
    print(f"📊 Graph saved as '{plot_file}'")
    plt.show()