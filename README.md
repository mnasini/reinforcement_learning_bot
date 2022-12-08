


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

<img width="293" alt="download (4)" src="https://user-images.githubusercontent.com/74600499/206561289-114655c1-d299-468d-826e-2d3235c1ee81.png">

# case study: approach for screen environment

let's now think of a situation in wich the environment is now TextFlappyBird-screenv0.<br>
This environment at each step returns the complete screen render of the
game, this differs much from the situation we had previously, in wich the environment only returned the x,y distance from the gap, infact with those information we are able to use traditional reinforcement learning techniques (q-learning , sarsa ,ecc) because the state space is discretisable and not so big (for a 15x20 grid we have 15x20x2=600 state,action(q) values, so this is feasible with simple exploration.<br>
In TextFlappyBird-screenv0 instead the observation returned is the entire screen render, this makes the q-value function henormous(all the possible pixel combinations. As is written before a traditional reinforcement learning algorithm (for example q-learning) is very slow in convergence, and, as long as we dont have days of running time to spare, this solution is unfeasible.<br>
To solve this problem we need to approximate the q-value function.<br>
A possible algorithm that make this kind of approximation is deep q-learning.
<br>it is common to use a function approximator to estimate the action-value function,
Q(s, a, w) ≈ Q∗
(s, a).typically a linear linear function approximator,is used but sometimes a non-linear function approximator can be used as well in the form of a neural network.<br>
In deep Q-learning, we use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output, or in case of a continuous action space, it can return the parameters of a distribution from wich we can sample the next action.<br>
the loss function is given by mean square error between the prediction and a target, meaning : $(r+γmax_{a'}Q(s',a',w'_i)-Q(s,a,w_i))^2$ in wich the target is $r+γmax_{a'}Q(s',a',w'_i)$ and the prediction is clearly $Q(s,a,w_i)$ .
Since the same network is calculating the predicted value and the target value, there could be a lot of divergence between these two. So, instead of using one neural network for learning, we can use two.
We could use a separate network to estimate the target. This target network has the same architecture as the function approximator but with frozen parameters. For every X iterations, the parameters from the prediction network are copied to the target network.<br> So the learning algoritm is the following: 
* use target network for predicting target (forward pass on target network)
* do a forward pass on prediction network as well(main network) 
* backpropagate loss trought the layers of the main network (updating its weights)
* after a fixed number of iterations copy the weights of the main network and substitute the weights of the target network with them (we are transferring the learning after some iteration)
* extract next action using policy(e.g ϵ-greedy) <br>
in our specific case the input is the screen vector in wich at each element can be @ for our bird ^ for the upper border [ for the with border and | for the pipe pixel, this vector will constitute the input state of the network.<br>
In case of the original environment(https://github.com/Talendar/flappy-bird-gym)  the observation at each step will be the screenshot image of the current state, the network architecture we should use would instead be a convolutional neural network, and the training steps would be exactly the same(only the first n linear layers block would be replaced by convolutional layers.

#comparisons of the two agents

the graph above shows the sums of reward after x episodes averaged over 100 runs to reduce stochasticity in the plot.<br>
since the game played is flappy bird and the agent will receive a reward as long as it remains alive, we clearly see that there isn't an upper bound of how much reward it can get, so is pretty normal that the sum of rewards will constantly increase with the number of episodes.<br>
As written below we know for sure that the q-learning algorithm is optimal as long as the number of episodes tend to infinite,while sarsa results are suboptimal, so , why in this graph it seems that the epected sarsa performs better? simply because the q-learning algorithm is very slow in convergence, and it needs to do a lot of more exploration before it finds the optimal policy.


<img width="269" alt="download (5)" src="https://user-images.githubusercontent.com/74600499/206561702-3e348e5f-3d7a-4f7d-849d-ff2d9c1d8fef.png">

this plot represents the mean sum of reward per episode with different step sizes.<br>The number of episode per run is 700(so more episodes to let q-learning converge).<br>
Q-learning (and off-policy learning in general) has higher per-episode variance than SARSA, and may suffer from problems converging as a result. but we can see that it will get a higher per sum of rewards if we are using more episodes, whatever is the step size, but if we keep a lower number of episodes, we can see that expected sarsa may have a better result<br><br>

## differencies of the two agents 2: 

expected sarsa is not being very optimistic because it does not always consider the best future value, instead it considers an outcome based on the current policy. 
sarsa will approach convergence allowing for possible penalties from exploratory moves, whilst q-learning will ignore them. That makes sarsa more conservative - if there is risk of a large negative reward close to the optimal path, q-learning will tend to trigger that reward whilst exploring, whilst sarsa will tend to avoid a dangerous optimal path and only slowly learn to use it when the exploration parameters are reduced. thats the main difference, thats why using sarsa can be extremely useful if, we had an envronment in wich we can have large expenses if we loose (e.g automatic driving) or if, in our case we would pay 1€ every time the bird loses, q-learnign, will try to find the optimal move (even pulling some dangerous actions to get exacly in the middle of the pipe, while sarsa is only interested in passing the pipe)



