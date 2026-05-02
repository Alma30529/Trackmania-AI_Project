from env import TrackmaniaPixelEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os

def main():
    print("="*50)
    print("🏎️  TRACKMANIA AI - PPO CNN TRAINING  🏎️")
    print("="*50)
    
    # 1. Initialize Environment
    env = TrackmaniaPixelEnv()
    
    # Sanity check to make sure the environment is valid
    print("Checking environment...")
    check_env(env)
    print("Environment OK!")

    # 2. Setup the PPO Agent with a CNN Policy
    # We use a lower learning rate because pixels are noisy
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=64,
        tensorboard_log="./trackmania_tensorboard/"
    )

    print("\nTraining starting in 3 seconds. CLICK INTO TRACKMANIA NOW!")
    import time
    time.sleep(3)

    # 3. Train the Agent
    try:
        # 100,000 steps takes a few hours, but you should see it learning 
        # to stop hitting walls within the first 10,000 steps.
        model.learn(total_timesteps=100000)
    except KeyboardInterrupt:
        print("\nTraining stopped manually.")
    finally:
        # Release all keys just in case
        env.keyboard.release('w')
        env.keyboard.release('a')
        env.keyboard.release('d')
        
        model.save("trackmania_cnn_agent")
        print("✅ Model saved as 'trackmania_cnn_agent.zip'")

if __name__ == "__main__":
    main()