


#  <br><b>Flappy bot, a reinforcement learning personal project</b><br>

the first agent implemented is the expected sarsa:<br> 
we can distinguish two main families of reinforcement learning techniques:
* On Policy: In this, the learning agent learns the value function according to the current action derived from the policy currently being used.
* Off Policy: In this, the learning agent learns the value function according to the action derived from another policy.<br>
sarsa is an on policy technique since it uses the action performed by the current policy to learn the Q-value.<br>
This following is the update equation for sarsa: $Q(s_t,a_t)=Q(s_t,a_t)+α(r_{t+1}+γQ(s_{t+1},a_{t+1})-Q(s_t,a_t))$ where $r_t$ is the reward at time t, $γ$ is the discount factor and $α$ is the learning rate.
<br> This update equation of this particular algorithm depends on the current state, next state and next action, we can actually write a sarsa that takes in account more steps, in that case it can be considered an n-step sarsa.
<br> expected sarsa is a slight variation of the sarsa algorithm: <br> we know infact that both our agents implemented (expected sars and (we'll see in the future) q-learning are algorithms that uses temporal difference updates to improve the agent behaviour.
<br> We know that sarsa is an on-policy technique, Q-learning is an off-policy technique, but expected sarsa can be use either as an on-policy or off-policy.
<br> Let's now compare the expected sarsa update equation with the sarsa update equation:
<b> expected sarsa:</b> $Q(s_t,a_t)+α(r_{t+1}+γ∑_aπ(a|s_{t+1}Q(s_{t+1},a)-Q(s_t,a_t))$ <br>
<b>sarsa: </b> $Q(s_t,a_t)=Q(s_t,a_t)+α(r_{t+1}+γQ(s_{t+1},a_{t+1})-Q(s_t,a_t))$ <br>
as is possible to see  expected sarsa takes the weighted sum of all possible next actions with respect to the probability of taking that action. If the Expected Return is greedy with respect to the expected return, then this equation gets transformed to Q-Learning.
<br> Importantly, because Sarsa is on-policy, it will not converge to optimal Q
values as long as exploration occurs. However, by annealing
exploration over time, Sarsa can converge to optimal Q
values, just like Q-learning, in my implementation i did't use an adaptive learning rate, because i wanted to compare learning rates in the two algorithms in the next sections
## analysis of the first agent
following is the plot of the state value function for the expected sarsa algorithm


<img width="643" alt="download" src="https://user-images.githubusercontent.com/74600499/206560179-0024b324-ac74-4578-b842-ee071f175e48.png">
the graph plots the final value of all states, the x axes shows the horizontal distance of the bird from the gap, the y axes shows the vertical distances of the bird from the gap, the z axes shows the value corresponding to the state (horizontal distance, vertical distance).<br>
theoretically we want the state value function to be as high as possible for all states, because in this way the plot will tell us that we dont have some "unluky" state, that have a lower value compared to the rest, so as we increase the number of episodes, as is possible to see from the plot below, the value associated to each state will remain high:
<img width="648" alt="download (1)" src="https://user-images.githubusercontent.com/74600499/206560358-3f02319f-ec32-49a4-b385-3fbb8c92ef77.png">
however, as written below , the expected sarsa algorithm do not always converge to the optimal policy (see reasons below), to make it converge, we should decrease the learning rate with the training (as the number of episodes increases, the learning rate should decrease)
here is our final policy plotted

<img width="305" alt="download (2)" src="https://user-images.githubusercontent.com/74600499/206560526-598acb70-921f-4fc1-bdee-f1291261cde1.png">

for each state the little square color represent the action we should play if its green we should idle, if its gray we should flap. For example we can see that we should flap almost all the time if the player current vertical distance from the gap is below 0 and idle almost all the time if the vertical distance is above 0 to play the game optimally. We can compare this plot as a rulebook that tells us what are the actions to take for each possible state.
lets now define the second agent(q-learning agent)
# implementation of the second agent
Q-learning is an off policy reinforcement learning algorithm that seeks to find the best action to take given the current state. It’s considered off-policy because the q-learning function learns from actions that are outside the current policy($ϵ-greedy)$ and therefore a policy isn’t needed. More specifically, q-learning seeks to learn a policy that maximizes the total reward(greedy policy).
the update rule is the following: 
$Q(s,a)=(1-α)Q(s,a)+\alpha(r+max_{a'}(Q(s',a'))$ where a' is the action chosen with the gredy policy, and a is rhe action choosen at each step ($ϵ-greedy)$, we clearly see that we are learning from a behaviour policy(ϵ-greedy) and we are evaluating our target policy (greedy) allowing them to improve at each step (td-learning). q-learning always converges to the optimal solution because is an off-policy learning algorithm. We are following the behaviour policy, $μ$, which is ϵ−greedy. This behaviour policy need not be an optimal policy rather it is a more explorable policy. But we are learning the target policy, π, which is argmax of state action value (Q(s,a)). This target policy is by definition optimal policy. however q -learning converges only given an adeguate exploration, in this case thee state space was pretty small (15x20x2) total possible states, so training for 500/1000 episodes was sufficient, but this is not the case if we used the original screen environment, in that case the state space would be enormous, and to use this algorithm we would need too much episodes, and this solution is not timely feasible.<br>
In that particular case i would have simply approximated the q-value function using a deep-q-learning algorithm(see next section)
<img width="568" alt="download (3)" src="https://user-images.githubusercontent.com/74600499/206560976-fd9b96ed-e975-4fef-a2cc-611c7bcc1fc5.png">
this is the state value function plotted for the q learning agent,as is possible to see there are a lot of more spike and uncertanty in values, but we can clearly see that is in the way of finding the optimal policy.
lets now train this too for 2000 episodes and plot its policy


