import gymnasium as gym
from stable_baselines3 import PPO
from PIL import Image

class FuelPenaltyWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if action == 2:
            reward -= 50.0
        return obs, reward, terminated, truncated, info

print("--- ENTRAÎNEMENT DE L'AGENT RADIN ---")
base_env = gym.make("LunarLander-v3")
train_env = FuelPenaltyWrapper(base_env)
model_cheap = PPO("MlpPolicy", train_env, verbose=1, device="cpu")
model_cheap.learn(total_timesteps=150000)
train_env.close()
print("Entraînement terminé.")

print("\n--- ÉVALUATION ET TÉLÉMÉTRIE ---")
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
obs, info = eval_env.reset()
done = False
frames = []
total_reward = 0.0
main_engine_uses = 0
side_engine_uses = 0

while not done:
    action, _states = model_cheap.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    if action == 2:
        main_engine_uses += 1
    elif action in [1, 3]:
        side_engine_uses += 1
    frames.append(Image.fromarray(eval_env.render()))
    done = terminated or truncated

eval_env.close()

if reward == -100:
    issue = "CRASH DÉTECTÉ 💥"
elif reward == 100:
    issue = "ATTERRISSAGE RÉUSSI 🏆"
else:
    issue = "TEMPS ÉCOULÉ OU SORTIE DE ZONE ⚠️"

print("\n--- RAPPORT DE VOL PPO HACKED ---")
print(f"Issue du vol : {issue}")
print(f"Récompense totale cumulée : {total_reward:.2f} points")
print(f"Allumages moteur principal : {main_engine_uses}")
print(f"Allumages moteurs latéraux : {side_engine_uses}")
print(f"Durée du vol : {len(frames)} frames")

if frames:
    frames[0].save('TP5/hacked_agent.gif', save_all=True, append_images=frames[1:], duration=30, loop=0)
    print("Vidéo sauvegardée sous 'TP5/hacked_agent.gif'")
