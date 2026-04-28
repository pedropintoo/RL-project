
Looking at the raw numbers, **the undisputed winner is $\beta = 0.1$.**.
Let's break down the results for each beta value and understand the underlying dynamics:

### 1. ($\beta = 2.0$)
At $\beta = 2.0$, the KL penalty is completely overpowering the Reward Model. 
* **CartPole:** Performance actually *crashed* down to ~100. It performs significantly worse than the Mid-Anchor (369). 
* **Pendulum:** Across all K values, the agent is stuck between -600 and -369. It is terrified of moving away from the Mid-Anchor (-644) because the math penalizes it too heavily for changing its behavior. 
* **Verdict:** Too conservative. The agent refuses to learn.

### 2. ($\beta = 0.5$)
This value allowed for *some* learning, but it still held the agent back.
* **CartPole:** It managed to hit 500 at K=200, but oddly collapsed at K=1000. 
* **Pendulum:** It hovered around -400. It successfully pulled away from the Mid-Anchor, but the penalty was still too high to let it reach the Expert level (-335).
* **Verdict:** Restrictive. It prevents the agent from fully optimizing the reward.

### 3. ($\beta = 0.01$)
With almost no KL penalty, the agent was completely unconstrained. It trusted the Reward Model blindly.
* **CartPole:** Very strong performance, hitting 497 at K=1000.
* **Pendulum:** Look closely at Pendulum K=50. The mean return is **-829.7**, which is dramatically *worse* than the Mid-Anchor (-627). 
* **Why did it fail at K=50?** This is classic **Reward Hacking**. At K=50, your Reward Model doesn't fully understand the physics yet, so it has "bugs" or "loopholes." Because $\beta = 0.01$ provides no safety leash, the PPO agent aggressively exploited those loopholes, breaking the actual game in the process. Once K=1000 provided a perfect Reward Model, the agent did great (-177), but it was highly unstable before that.
* **Verdict:** Too reckless. Highly vulnerable to exploiting poorly trained reward models.

### 4. ($\beta = 0.1$) - THE WINNER
This value struck the perfect mathematical balance.
* **CartPole:** Flawless 500s across the board for K=200 and K=1000.
* **Pendulum:** It achieved superhuman performance consistently (-240 at K=50, scaling beautifully to -187 at K=1000). 
* **Why it worked:** The leash was loose enough to let the agent surpass the Expert, but tight enough to prevent the catastrophic Reward Hacking we saw in $\beta = 0.01$ at K=50.

Now we have empirical evidence to show that $\beta = 0.1$ is the optimal choice for our main experiments!