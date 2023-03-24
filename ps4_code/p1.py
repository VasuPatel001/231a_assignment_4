import numpy as np

class Q1_solution(object):

  @staticmethod
  def system_matrix():
    """ Implement the answer to Q1A here.
    Output:
      A: 6x6 numpy array for the system matrix.
    """
    A = np.eye(6)
    # TODO fill in values for A below
    # Hint: only 9 entries should be non-zero, with values 1, 0.1, and 0.8
    # x = [px, py, pz, vx, vy, vz]
    dt = 0.1
    v = 0.8
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt
    A[3, 3] = v
    A[4, 4] = v
    A[5, 5] = v
    print(A)
    return A

  @staticmethod
  def process_noise_covariance():
    """ Implement the covariance matrix Q for process noise.
    Output:
      Q: 6x6 numpy array for the covariance matrix.
    """
    Q = np.eye(6)
    # TODO fill in values for Q below
    # Hint: only 3 entries should be non-zero, and all should equal 0.05
    Q[0, 0] = 0.00
    Q[1, 1] = 0.00
    Q[2, 2] = 0.00
    Q[3, 3] = 0.05
    Q[4, 4] = 0.05
    Q[5, 5] = 0.05
    print(Q)
    return Q

  @staticmethod
  def observation_noise_covariance():
    """ Implement the covariance matrix R for observation/measurement noise.
    Output:
      R: 2x2 numpy array for the covariance matrix.
    """
    R = np.eye(2)
    # TODO fill in values for R below
    # Hint: only 2 entries should be non-zero, and all should equal 5.0
    R = np.diag([5.0, 5.0])
    return R

  @staticmethod
  def observation(state):
    """ Implement the function h, from state to noise-less observation. (Q1B)
    Input:
      state: (6,) numpy array representing state.
    Output:
      obs: (2,) numpy array representing observation.
    """
    # Hint: you should use the camera intrinsics here
    K = np.array([[500, 0, 320, 0],
                 [0, 500, 240, 0],
                 [0, 0, 1, 0]])
    x, y, z = state[0], state[1], state[2]
    hom_3d = np.array([x, y, z, 1])
    hom_2d = K.dot(hom_3d).T
    point_2d = hom_2d / hom_2d[2]
    obs = np.array([point_2d[0], point_2d[1]])
    return obs

  def simulation(self, T=100):
    """ simulate with fixed start state for T timesteps.
    Input:
      T: an integer (=100).
    Output:
      states: (T,6) numpy array of states, including the given start state.
      observations: (T,2) numpy array of observations, Including the observation of start state.
    Note:
      We have set the random seed for you. Please only use np.random.multivariate_normal to sample noise.
      Keep in mind this function will be reused for Q2 by inheritance.
    """
    x_0 = np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0])
    states = [x_0]
    A = self.system_matrix()
    Q = self.process_noise_covariance()
    R = self.observation_noise_covariance()
    z_0 = self.observation(x_0) + np.random.multivariate_normal(np.zeros((R.shape[0],)), R)
    observations = [z_0]
    mean = []
    for t in range(1,T):
        process_noise = np.random.multivariate_normal(np.zeros((Q.shape[0],)), Q)
        present_state = A.dot(states[t-1]) + process_noise

        observation_noise = np.random.multivariate_normal(np.zeros((R.shape[0],)), R)
        present_obs = self.observation(present_state) + observation_noise

        states.append(present_state)
        observations.append(present_obs)
    return np.array(states), np.array(observations)

  @staticmethod
  def observation_state_jacobian(x):
    """ Implement your answerc for Q1D.
    Input:
      x: (6,) numpy array, the state we want to do jacobian at.
    Output:
      H: (2,6) numpy array, the jacobian of the observation model w.r.t state.
    """
    H = np.zeros((2,6))
    # Hint: four values in the Jacobian should be non-zero
    tx = x[0]
    ty = x[1]
    tz = x[2]
    Jacobian = np.array([[500/tz, 0, 500*-tx/tz**2, 0, 0, 0],
                         [0, 500/tz, 500*-ty/tz**2, 0, 0, 0]])

    return Jacobian

  def EKF(self, observations):
    """ Implement Extended Kalman filtering (Q1E)
    Input:
      observations: (N,2) numpy array, the sequence of observations. From T=1.
      mu_0: (6,) numpy array, the mean of state belief after T=0
      sigma_0: (6,6) numpy array, the covariance matrix for state belief after T=0.
    Output:
      state_mean: (N,6) numpy array, the filtered mean state at each time step. Not including the
                  starting state mu_0.
      state_sigma: (N,6,6) numpy array, the filtered state covariance at each time step. Not including
                  the starting state covarance matrix sigma_0.
      predicted_observation_mean: (N,2) numpy array, the mean of predicted observations. Start from T=1
      predicted_observation_sigma: (N,2,2) numpy array, the covariance matrix of predicted observations. Start from T=1
    Note:
      Keep in mind this function will be reused for Q2 by inheritance.
    """
    mu_0 = np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0])
    sigma_0 = np.eye(6)*0.01
    sigma_0[3:,3:] = 0.0
    A = self.system_matrix()
    Q = self.process_noise_covariance()
    R = self.observation_noise_covariance()
    state_mean = [mu_0]
    state_sigma = [sigma_0]
    predicted_observation_mean = []
    predicted_observation_sigma = []
    for ob in observations:
        mu_bar_next = A.dot(state_mean[-1]) # TODO fill this in
        sigma_bar_next = A.dot(state_sigma[-1]).dot(A.T) + Q # TODO fill this in
        H = self.observation_state_jacobian(mu_bar_next) # TODO fill this in
        kalman_gain_numerator = sigma_bar_next.dot(H.T) # TODO fill this in
        kalman_gain_denominator = H.dot(kalman_gain_numerator) + R # TODO fill this in
        kalman_gain = kalman_gain_numerator.dot(np.linalg.inv(kalman_gain_denominator)) # TODO fill this in
        expected_observation = self.observation(mu_bar_next) # TODO fill this in
        mu_next = mu_bar_next + kalman_gain.dot(ob - expected_observation) # TODO fill this in
        sigma_next = (np.eye(6) - kalman_gain.dot(H)).dot(sigma_bar_next) # TODO fill this in

        state_mean.append(mu_next)
        state_sigma.append(sigma_next)
        predicted_observation_mean.append(expected_observation)
        predicted_observation_sigma.append(kalman_gain_denominator)
    return np.array(state_mean[1:]), np.array(state_sigma[1:]), np.array(predicted_observation_mean), np.array(predicted_observation_sigma)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d

    np.random.seed(402)
    solution = Q1_solution()
    states, observations = solution.simulation()
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(states[:,0], states[:,1], states[:,2], c=np.arange(states.shape[0]))
    plt.show()

    fig = plt.figure()
    plt.scatter(observations[:,0], observations[:,1], c=np.arange(states.shape[0]), s=4)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()

    observations = np.load('./data/Q1E_measurement.npy')
    filtered_state_mean, filtered_state_sigma, predicted_observation_mean, predicted_observation_sigma = \
        solution.EKF(observations)
    # plotting
    true_states = np.load('./data/Q1E_state.npy')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_states[:,0], true_states[:,1], true_states[:,2], c='C0')
    for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
        draw_3d(ax, cov[:3,:3], mean[:3])
    ax.view_init(elev=10., azim=30)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:,0], observations[:,1], s=4)
    for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        draw_2d(ax, cov, mean)
    plt.xlim([0,640])
    plt.ylim([0,480])
    plt.gca().invert_yaxis()
    plt.show()



