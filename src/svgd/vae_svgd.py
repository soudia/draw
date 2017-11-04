from svgd import SVGD
import tensorflow as tf


class VariationalEncSVGD(SVGD):
    ''' Variational Auto-encoder Stein Variational Gradient Descent '''
    def __init__(self, stepsize=1e-3, n_hidden=10, num_particles=10,
                 dim_particle=5, num_epoch=10):
        ''' Initializer
            Args:
                stepsize:
                n_hidden:
                num_particles:
                dim_particles:
                num_epoch:
        '''
        self.stepsize = stepsize
        self.n_hidden = n_hidden
        self.num_epoch = num_epoch
        self.num_particles = num_particles
        self.dim_particle = dim_particle

        super(VariationalEncSVGD, self).__init__()

    def dlnprob(self, theta):
        pass

    def _init_weights(self, name, shape, stddev=0.1):
        ''' Weight initialization '''
        weights = tf.random_normal(shape, stddev=stddev)
        return tf.Variable(weights, name)

    def _forwardprop(self, X, w_1, w_2):
        ''' Forward-propagation. '''
        h = tf.tensordot(X, w_1, axes=[[1], [0]])
        yhat = tf.tensordot(h, w_2, axes=[[1], [0]])
        return yhat

    def recognition_model(self, h_enc, lnprob, num_iter, eta_dim):
        ''' Recognition model
            Args:
                h_enc:
        '''
        particles = tf.random_normal(shape=(h_enc.shape[0].value,
                                            self.num_particles,
                                            self.dim_particle),
                                     stddev=1.0)

        w_1 = self._init_weights('w_1', (h_enc.shape[1].value + eta_dim,
                                         self.n_hidden,
                                         self.num_particles))

        w_2 = self._init_weights('w_2', (self.n_hidden, self.dim_particle))

        X = tf.placeholder(tf.float32, shape=[h_enc.shape[0].value,
                                              h_enc.shape[1] + eta_dim],
                           name='X')

        y = tf.placeholder(tf.float32, shape=[h_enc.shape[0].value,
                                              self.num_particles,
                                              self.dim_particle],
                           name='y')

        yhat = self._forwardprop(X, w_1, w_2)

        cost = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=y, predictions=yhat))

        optimizer = tf.train.AdamOptimizer(self.stepsize, beta1=0.5)
        updates = optimizer.compute_gradients(cost)

        noise = self._init_weights('noise', [h_enc.shape[0].value, eta_dim])
        noisy_h_enc = tf.concat([h_enc, noise], axis=1)

        particles = tf.tensordot(noisy_h_enc, w_1, axes=[[1], [0]])
        particles = tf.tensordot(particles, w_2, axes=[[1], [0]])

        # particles = self.update(particles, lnprob, n_iter=num_iter)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        for epoch in range(self.num_epoch):
            sess.run(updates, feed_dict={X: sess.run(noisy_h_enc),
                                         y: sess.run(particles)})
            loss = sess.run(cost, feed_dict={X: sess.run(noisy_h_enc),
                                             y: sess.run(particles)})
            print("Epoch = {}, recognition model loss = {:.7f}".format
                  (epoch + 1, 100. * loss))
        # TODO Compute the mean, the logsigma, and the stdev
        return particles


if __name__ == '__main__':
    x_1 = tf.random_normal([4, 3], stddev=0.01)
    vae_svgd = VariationalEncSVGD()
    vae_svgd.recognition_model(x_1, vae_svgd.dlnprob, 2, 2)
    # with tf.Session() as sess:
    #     x_1 = tf.random_normal([4, 3], stddev=0.01)
    #     sess.run(tf.global_variables_initializer())
    #     vae_svgd.recognition_model(x_1, vae_svgd.dlnprob, 2, sess)
