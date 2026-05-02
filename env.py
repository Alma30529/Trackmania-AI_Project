import gymnasium as gym  # Changed from import gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import time
from pynput.keyboard import Key, Controller

class TrackmaniaPixelEnv(gym.Env):
    def __init__(self):
        super(TrackmaniaPixelEnv, self).__init__()
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
        
        self.keyboard = Controller()
        self.sct = mss.mss()
        self.capture_region = {"top": 110, "left": 90, "width": 800, "height": 600}
        
        self.last_frame = None
        self.stuck_counter = 0
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Gymnasium requires handling seed
        super().reset(seed=seed)
        
        self.current_step = 0
        self.stuck_counter = 0
        self.last_frame = None
        
        self.keyboard.release(Key.up)
        self.keyboard.release(Key.left)
        self.keyboard.release(Key.right)
        
        self.keyboard.press(Key.enter)
        time.sleep(0.1)
        self.keyboard.release(Key.enter)
        
        print("🚗 Resetting track...")
        time.sleep(2.0)
        
        # Gymnasium reset must return (obs, info)
        return self._get_state(), {}

    def step(self, action):
        self.keyboard.press(Key.up)
        
        if action == 1:
            self.keyboard.press(Key.left)
            self.keyboard.release(Key.right)
        elif action == 2:
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.left)
        else:
            self.keyboard.release(Key.left)
            self.keyboard.release(Key.right)

        time.sleep(0.1)
        state = self._get_state()
        reward, done = self._compute_reward(state, action)
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Gymnasium step must return (obs, reward, terminated, truncated, info)
        # We use 'done' for both terminated and truncated here
        return state, reward, done, False, {}

    def _get_state(self):
        screen = self.sct.grab(self.capture_region)
        img = np.array(screen)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img = cv2.resize(img, (64, 64))
        return np.expand_dims(img, axis=-1)

    def _compute_reward(self, current_frame, action):
        reward = 0
        done = False
        if self.last_frame is not None:
            diff = np.abs(current_frame.astype(np.int16) - self.last_frame.astype(np.int16))
            change = np.mean(diff)
            if change < 2.0:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            reward += (change * 0.5)
            if action in [1, 2]:
                reward -= 1.0 
        
        self.last_frame = current_frame
        if self.stuck_counter >= 15:
            reward = -50
            done = True
        return reward, done