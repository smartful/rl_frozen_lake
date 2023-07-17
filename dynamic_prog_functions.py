import numpy as np


def getQValue(env, V, s, a, gamma) -> float:
    q = 0
    for (proba_s_prime, s_prime, reward_s_a, done) in env.P[s][a]:
        q += proba_s_prime * (reward_s_a + gamma * V[s_prime])

    return q


def evaluatePolicy(env, pi, V, gamma, k):
    V_updated = np.copy(V)
    improved = True

    for i in range(k):
        for s in range(env.nS):
            V_new = 0
            for a in range(env.nA):
                prob_a = pi[s][a]
                q_s_a = getQValue(env, V_updated, s, a, gamma)
                V_new += prob_a * q_s_a

            V_updated[s] = V_new

    if (np.array_equal(V, V_updated)):
        improved = False

    return V_updated, improved


def improvePolicy(env, pi, V, gamma):
    for s in range(env.nS):
        q_s = np.zeros([env.nA, 1])
        for a in range(env.nA):
            q_s[a] = getQValue(env, V, s, a, gamma)

        best_a = np.argmax(q_s)
        pi[s] = np.eye(env.nA)[best_a]

    return pi
