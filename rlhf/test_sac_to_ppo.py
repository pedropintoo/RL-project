import gymnasium as gym
from stable_baselines3 import SAC, PPO
import os

def run_crash_experiment():
    env_id = "MountainCarContinuous-v0"
    file_name = "mountaincar_mid_sac_dummy"

    print(f"=== Phase 1: Creating the SAC Model ===")
    env = gym.make(env_id)
    
    # Initialize a SAC model and do a single step just to generate the weights
    sac_model = SAC("MlpPolicy", env, verbose=0)
    sac_model.learn(total_timesteps=1)
    
    # Save the SAC model to disk
    sac_model.save(file_name)
    print(f"✅ Successfully saved SAC model to {file_name}.zip\n")

    print(f"=== Phase 2: The Forbidden Load ===")
    print("Attempting to run: PPO.load('mountaincar_mid_sac_dummy.zip')\n")
    
    try:
        # THIS IS THE LINE THAT WILL CRASH
        ppo_model = PPO.load(file_name, env=env, device="cpu")
        print("Wait, it succeeded?! (This should not print)")
        
    except Exception as e:
        print("💥 INSTANT CRASH 💥")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message:\n{e}\n")
        
        print("Why this happened:")
        print("Stable-Baselines3 checks the 'data' dictionary inside the .zip file.")
        print("It sees parameters for 'actor' and 'critic' (SAC's Q-networks),")
        print("but PPO is looking for 'policy' and 'value_net'. Because the dictionary")
        print("keys and tensor shapes don't match, PyTorch throws an error trying to map them.")

    # Clean up the dummy file so it doesn't clutter your directory
    if os.path.exists(f"{file_name}.zip"):
        os.remove(f"{file_name}.zip")
        print("Cleaned up dummy zip file.")

if __name__ == "__main__":
    run_crash_experiment()