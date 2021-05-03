import gym
#Chargez l’environnement frozen-lake-v0.
env = gym.make("FrozenLake-v0")
#Mettre l'environnement dans son état initial
env.reset()
#Afficher l’environnement
env.render()

print("espace des observations", env.observation_space)
print("espace d'actions", env.action_space)
# affichez l’ensemble des probabilités d’évolutions
print(env.P)
# 10 d’actions tirées au hasard et affichage de  l’environnement et des données reçues.
MAX_ITERATIONS = 10

env.render()
for i in range(MAX_ITERATIONS):
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(
        random_action)
    env.render()
    print("action", new_state)
    print("etat",new_state)
    print("Gain", reward)
    print("Terminé", done)
    print("Debug", info)
    if done:
        break

# Ecrire la suite d’instructions qui permet d’effectuer une suite d’actions et de collecter les récompenses jusqu’à ce que l’on tombe dans un trou.
env.reset()
print("---------------------------------suite d'actions-------------------------------------------")
bool = True

gainTotal = 0
while(bool):
    random_action = env.action_space.sample()
    print("Action:", random_action)
    new_state, reward, done, info = env.step(random_action)  # L'agent qui fait une action aléatoire, done est le cas ou il tombe dans le trou**
    print("obser", new_state)
    gainTotal +=reward
    print("gain",gainTotal)
    env.render()  # affichage de l'agent
    if done:
        break

# Ecrire la fonction qui implémente l’amélioration ε greedy.
print("Fonction test ε greedy.")
def test_performance(policy, nb_episodes=100):
    sum_returns = 0
    for i in range(nb_episodes):
        state  = env.reset()
        done = False
        while not done:
            action = policy(state)
            state, reward, done, info = env.step(action)
            if done:
                sum_returns += reward
    return sum_returns/nb_episodes

policy_dict = {0:1, 1:2, 2:1, 3:0, 4:1, 6:1, 8:2, 9:0, 10:1, 13:2, 14:2} #random policy
policy = lambda s: policy_dict[s]
print("politique",test_performance(policy))
# Ecrire un algorithme de Q learning.