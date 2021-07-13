from .source.mmd_ma import *

# 'Change:' comments are changes for this wrapper
def mmd_ma_helper(k1_matrix, 
                  k2_matrix,
                  tradeoff2=.01,
                  tradeoff3=.001,
                  p=2,
                  bandwidth=1.0,
                  training_rate=.00005,
                  k=0
                  ):
    I_p=tf.eye(p)
    record = open('loss.txt', 'w')
    n1 = k1_matrix.shape[0]
    n2 = k2_matrix.shape[0]
    K1 = tf.constant(k1_matrix, dtype=tf.float32)
    K2 = tf.constant(k2_matrix, dtype=tf.float32)
    # Changed 'random_' to 'random.'
    alpha = tf.Variable(tf.random.uniform([n1,p],minval=0.0,maxval=0.1,seed=k))
    beta = tf.Variable(tf.random.uniform([n2,p],minval=0.0,maxval=0.1,seed=k))

    # Changed into function form
    def cost():
        mmd_part = maximum_mean_discrepancy(tf.matmul(K1,alpha), tf.matmul(K2,beta), bandwidth=bandwidth)
        penalty_part = tradeoff2*(tf.norm(tf.subtract(tf.matmul(tf.transpose(alpha),tf.matmul(K1,alpha)),I_p),ord=2) + tf.norm(tf.subtract(tf.matmul(tf.transpose(beta),tf.matmul(K2,beta)),I_p),ord=2))
        distortion_part = tradeoff3*(tf.norm(tf.subtract(tf.matmul(tf.matmul(K1,alpha),tf.matmul(tf.transpose(alpha),tf.transpose(K1))),K1),ord=2)+tf.norm(tf.subtract(tf.matmul(tf.matmul(K2,beta),tf.matmul(tf.transpose(beta),tf.transpose(K2))),K2),ord=2))
        myFunction = mmd_part + penalty_part + distortion_part
        return myFunction
    # Changed 'train.AdamOptimizer' to 'optimizers.Adam'
    # Changed 'myFunction' to 'cost'
    # Added var_list=[alpha, beta]
    train_step = tf.optimizers.Adam(training_rate).minimize(cost, var_list=[alpha, beta])
    
    # Added to disable eager execution
    tf.compat.v1.disable_eager_execution()
    
    # Changed for tf2, removed
    #init = tf.global_variables_initializer()
    #sess = tf.Session()
    #sess.run(init)
    for i in range(10001):
      sess.run(train_step)
      if (i%100 == 0): 
        np.savetxt("alpha_hat_"+str(k)+"_"+str(i)+".txt", sess.run(alpha))
        np.savetxt("beta_hat_"+str(k)+"_"+str(i)+".txt", sess.run(beta))
        rec = '\t'.join([str(k), str(i), str(sess.run(myFunction)), str(sess.run(mmd_part)), str(sess.run(penalty_part)), str(sess.run(distortion_part))])  
        record.write(rec + '\n')
        #print i
        #print(sess.run(myFunction))
