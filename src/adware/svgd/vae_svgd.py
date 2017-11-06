''' Variational Auto-encoder Stein Variational Gradient Descent '''

from svgd.svgd import SVGD
import tensorflow as tf


# pylint: disable=E1129
class VariationalEncSVGD(SVGD):
    ''' Variational Auto-encoder Stein Variational Gradient Descent '''
    def __init__(self, stepsize=1e-3, n_hidden=10, num_epoch=10):
        ''' Initializer
            Args:
                stepsize:
                n_hidden:
                num_epoch:
        '''
        self.stepsize = stepsize
        self.n_hidden = n_hidden
        self.num_epoch = num_epoch

        tf.set_random_seed(12345)

        super(VariationalEncSVGD, self).__init__()

    def update(self, theta, gradients, n_iter):
        def dlnprob(theta):
            return gradients

        # theta = super(VariationalEncSVGD, self).update(theta, dlnprob, n_iter=n_iter)
        return theta

    def _init_weights(self, name, shape, stddev=0.1):
        ''' Weight initialization '''
        with tf.name_scope("%s_init" % name):
            weights = tf.random_normal(shape, stddev=stddev)
        return tf.Variable(weights, name)

    def _forwardprop(self, X, w_1, w_2):
        ''' Forward-propagation. '''
        h = tf.tensordot(X, w_1, axes=[[1], [0]])
        yhat = tf.tensordot(h, w_2, axes=[[1], [0]])
        return yhat

    def recognition_model(self, noisy_h_enc, particles, num_iter, time_step):
        ''' Recognition model
            Args:
                h_enc:
        '''
        num_particles = particles.shape[1].value
        dim_particle = particles.shape[2].value

        w_1 = self._init_weights('w_1', (noisy_h_enc.shape[1].value,
                                         self.n_hidden,
                                         num_particles))

        w_2 = self._init_weights('w_2', (self.n_hidden, dim_particle))

        # X = tf.placeholder(tf.float32, shape=[noisy_h_enc.shape[0].value,
        #                                       noisy_h_enc.shape[1].value],
        #                    name='X')

        # y = tf.placeholder(tf.float32, shape=[noisy_h_enc.shape[0].value,
        #                                       num_particles,
        #                                       dim_particle],
        #                    name='y')

        yhat = self._forwardprop(noisy_h_enc, w_1, w_2)

        with tf.name_scope("metrics"):
            cost = tf.reduce_mean(tf.squared_difference(particles, yhat),
                                  name='mse')

        with tf.variable_scope("optimization_", reuse=None):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.stepsize,
                                               name=time_step)
            gradients = optimizer.compute_gradients(cost)
            train_op = optimizer.apply_gradients(gradients, global_step=None,
                                                 name='train_op_')

        # noise = self._init_weights('noise', [h_enc.shape[0].value, eta_dim])
        # noisy_h_enc = tf.concat([h_enc, noise], axis=1)

        # sess = tf.InteractiveSession()
        # sess.run(tf.global_variables_initializer())

        # for epoch in range(self.num_epoch):
        #     _, loss = sess.run([train_op, cost],
        #                        feed_dict={X: sess.run(noisy_h_enc),
        #                                   y: sess.run(particles)})

        #     print("Epoch = {}, recognition model loss = {:.7f}".format
        #           (epoch + 1, 100. * loss))
        #     print(sess.run(gradients,
        #                    feed_dict={X: sess.run(noisy_h_enc),
        #                               y: sess.run(particles)}))

            particles = tf.tensordot(noisy_h_enc, w_1, axes=[[1], [0]])
            particles = tf.tensordot(particles, w_2, axes=[[1], [0]])
            particles = self.update(particles, gradients, num_iter)

        return particles, train_op


if __name__ == '__main__':
    x_1 = tf.random_normal([4, 3], stddev=0.01)
    particles_ = tf.random_normal(shape=(x_1.shape[0].value,
                                         10, 5), stddev=1.0)
    vae_svgd = VariationalEncSVGD()
    vae_svgd.recognition_model(x_1, particles_, 2, 2)
    # with tf.Session() as sess:
    #     x_1 = tf.random_normal([4, 3], stddev=0.01)
    #     sess.run(tf.global_variables_initializer())
    #     vae_svgd.recognition_model(x_1, vae_svgd.dlnprob, 2, sess)
