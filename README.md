# Intelligent Space Invaders game

Our deep Q neural network takes a stack of 4 frames as an input. THese pass through its network and output a vector of Q values for each action possible in the given state.
In the begining the frame does really bad but overtime it begins to associate frames(states) with the best actions to do.
Next is preprocessing which is an important step to reduce the complexity of our states and to reduce the computation time needed in training. 4 frames have been used to reduce the problem of temporal limitation. ie. If we give a frame at a time it has no idea of motion.
It uses convolutional networks where frames are processed by 3 convolution layers each using ELU as an activation function that produces the Qvalue estimation for each action

Calculates next state to process using the bellman equation

# Algorithm
  <pre>
  <br />Initialise Space Invaders environment
  <br />Initialise replay memory M with capacity N
  <br />Inisialise the DQN weights
  <br />for episode in max episode
    <br />s = environment state
    <br />for steps in maxsteps
      <br />choose action a from state using epsilon greedy
      <br />Take action a, get r(reward) and s(next state)
      <br />Store experience tuple<a,a,r,s> in M
      <br />s = s'(state = newstate)
      <br />Get random minibatch of exp tuples from M
      <br />Set Q_target = reward(s,a) + ymaxQ('s')
      <br />update w= alpha(Qtarget - Qvalue)*change in Qvalue
  </pre>
