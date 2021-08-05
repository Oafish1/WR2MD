from .source.mmd_ma import *
import tensorflow.compat.v1 as tf

# Change: Adding compatibility for raw data
from scipy.spatial.distance import cdist

# 'Change:' comments are changes for this wrapper
def mmd_ma_helper(k1_matrix, 
                  k2_matrix,
                  tradeoff2=.01,
                  tradeoff3=.001,
                  p=2,
                  bandwidth=1.0,
                  training_rate=.00005,
                  k=0,
                  # Change: Added variable stat recording timings
                  rec_step=100,
                  # Change: Added max iterations
                  max_iterations=10001,
                  # Change: Added compatibility for raw data
                  similarity_matrices=True
                  ):
    """
    Performs MMD-MA

    Parameters
    ----------
    *args: args
        Any args to use while running the specified method
    **kwargs: kwargs
        Any parameters to use while running the specified method
    rec_step: int
        How often to record statistics into the 'history' dictionary
    max_iterations: int
        Number of iterations to run
    similarity_matrices: bool
        If false, calculates similarity matrices on input to use
        in the algorithm

    Returns
    -------
    (
        [Mapped datasets],
        Dictionary object with 'loss' (, _mmd, _penalty, _distortion), 
        'alpha', 'beta', 'iteration' entries,
        [Solved weights]
    )
    """

    # Change: Add compatibility for raw data
    if not similarity_matrices:
        #k1_matrix = cdist(k1_matrix, k1_matrix, metric='euclidean')
        #k2_matrix = cdist(k2_matrix, k2_matrix, metric='euclidean')
        k1_matrix = (k1_matrix - np.mean(k1_matrix, 0)) / np.std(k1_matrix, 0)
        k1_matrix = np.matmul(k1_matrix, k1_matrix.T)
        k2_matrix = (k2_matrix - np.mean(k2_matrix, 0)) / np.std(k2_matrix, 0)
        k2_matrix = np.matmul(k2_matrix, k2_matrix.T)

    # Change: Compatibility
    tf.disable_eager_execution()
    #tf.compat.v1.enable_eager_execution()
    
    # Change: Add history dict for statistic storage
    history = {'loss': [], 
               'loss_mmd': [], 
               'loss_penalty': [], 
               'loss_distortion': [], 
               'alpha': [],
               'beta': [], 
               'iteration': []
               }
    
    I_p=tf.eye(p)
    # Change: Removed
    #record = open('loss.txt', 'w')
    n1 = k1_matrix.shape[0]
    n2 = k2_matrix.shape[0]
    K1 = tf.constant(k1_matrix, dtype=tf.float32)
    K2 = tf.constant(k2_matrix, dtype=tf.float32)
    alpha = tf.Variable(tf.random_uniform([n1,p],minval=0.0,maxval=0.1,seed=k))
    beta = tf.Variable(tf.random_uniform([n2,p],minval=0.0,maxval=0.1,seed=k))

    # myFunction = tradeoff1*maximum_mean_discrepancy(tf.matmul(K1,alpha), tf.matmul(K2,beta)) + tradeoff2*(tf.norm(tf.subtract(tf.matmul(tf.transpose(alpha),tf.matmul(K1,alpha)),I_p),ord=2) + tf.norm(tf.subtract(tf.matmul(tf.transpose(beta),tf.matmul(K2,beta)),I_p),ord=2)) + tradeoff3*(tf.norm(tf.subtract(tf.matmul(tf.matmul(K1,alpha),tf.matmul(tf.transpose(alpha),tf.transpose(K1))),K1),ord=2)+tf.norm(tf.subtract(tf.matmul(tf.matmul(K2,beta),tf.matmul(tf.transpose(beta),tf.transpose(K2))),K2),ord=2))
    mmd_part = maximum_mean_discrepancy(tf.matmul(K1,alpha), tf.matmul(K2,beta), bandwidth=bandwidth)
    penalty_part = tradeoff2*(tf.norm(tf.subtract(tf.matmul(tf.transpose(alpha),tf.matmul(K1,alpha)),I_p),ord=2) + tf.norm(tf.subtract(tf.matmul(tf.transpose(beta),tf.matmul(K2,beta)),I_p),ord=2))
    distortion_part = tradeoff3*(tf.norm(tf.subtract(tf.matmul(tf.matmul(K1,alpha),tf.matmul(tf.transpose(alpha),tf.transpose(K1))),K1),ord=2)+tf.norm(tf.subtract(tf.matmul(tf.matmul(K2,beta),tf.matmul(tf.transpose(beta),tf.transpose(K2))),K2),ord=2))
    myFunction = mmd_part + penalty_part + distortion_part
    train_step = tf.train.AdamOptimizer(training_rate).minimize(myFunction)

    init = tf.global_variables_initializer()
    # Change: Prevent leaks
    with tf.Session() as sess:
        sess.run(init)
        for i in range(max_iterations):
          sess.run(train_step)
          if (i%rec_step == 0): 
            # Change: Replaced
            #np.savetxt("alpha_hat_"+str(k)+"_"+str(i)+".txt", sess.run(alpha))
            #np.savetxt("beta_hat_"+str(k)+"_"+str(i)+".txt", sess.run(beta))
            history['iteration'].append(i)
            # Maybe remove these logs?  Storage?
            history['alpha'].append(sess.run(alpha))
            history['beta'].append(sess.run(beta))
            # Change: Replaced
            #rec = '\t'.join([str(k), str(i), str(sess.run(myFunction)), str(sess.run(mmd_part)), str(sess.run(penalty_part)), str(sess.run(distortion_part))]) 
            #record.write(rec + '\n')
            history['loss'].append(sess.run(myFunction))
            history['loss_mmd'].append(sess.run(mmd_part))
            history['loss_penalty'].append(sess.run(penalty_part))
            history['loss_distortion'].append(sess.run(distortion_part))
            #print i
            #print(sess.run(myFunction))
        
        map_K1 = tf.matmul(K1,alpha).eval(session=sess)
        map_K2 = tf.matmul(K2,beta).eval(session=sess)
    # Change: Added return
    return ([map_K1, map_K2], history, [alpha, beta])
