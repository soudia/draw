""" Queue Runner """
import tensorflow as tf
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer


class QueueReader(object):
    """ Create  batches that it stores in a Tensor flow queue for easy consumption. """
    def __init__(self, data_dir, batch_size=128, max_queue_size=128, z_dim=100, sess=tf.Session()):
        """ Initializer
            :param data_dir: Directory that contains the training files (absolute path)
            :param batch_size: Size of a batch
            :param max_queue_size: Maximum size of the queue
        """
        # sess.graph._unsafe_unfinalize()
        self.embeddings = self._create_embeddings(data_dir, batch_size, z_dim)
        self.queue = self._build_queue(self.embeddings, max_queue_size)
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

    def dequeue_many(self, num_elements):
        '''
        :param num_elements: number of items to be dequeued
        returns a tensorflow qeueue.dequeue_many operation
        '''
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue(self):
        '''
        returns a tensorflow qeueue.dequeue operation
        '''
        return self.queue.dequeue()

    def _build_queue(self, embeddings, max_queue_size):
        '''
        :param z_dim: Dimensionality of the latent space
        :params embeddings: Latent codes aka embeddings
        '''
        # max_queue_size = embeddings.shape[0].value * embeddings.shape[1].value
        queue = tf.FIFOQueue(capacity=max_queue_size,
                             shapes=([embeddings.shape[0].value,
                                      embeddings.shape[1].value,
                                      embeddings.shape[2].value]),
                             dtypes=(tf.float32), names=('inputs'))

        # embeddings = tf.reshape(embeddings, shape=[
        #     max_queue_size, embeddings.shape[2].value])

        enqueue = queue.enqueue({'inputs': embeddings})

        num_threads = 1

        runner = tf.train.QueueRunner(queue, [enqueue] * num_threads)
        tf.train.add_queue_runner(runner)
        return queue

    def _create_embeddings(self, data_dir, batch_size, z_dim):
        '''
        :param data_dir: Directory that contains the training files (absolute path)
        :param batch_size: Size of a batch
        returns an embedding layer
        '''
        data_dict = load_files(data_dir)

        vectorizer = CountVectorizer()
        vocabulary = vectorizer.fit_transform(data_dict['data'])
        vocabulary = vectorizer.get_feature_names()
        analyzer = vectorizer.build_analyzer()

        indices = []

        for data in data_dict['data']:
            indices.extend([vocabulary.index(tok) for tok in analyzer(data)])
        z = tf.get_variable('z', (len(indices), z_dim), tf.float32)
        embeddings = tf.gather(z, indices)
        num_batch = embeddings.shape[0].value // batch_size
        if (embeddings.shape[0] % batch_size > 0):
            num_batch = num_batch + 1
        flatten = tf.reshape(embeddings, [-1])
        zero_padding = tf.zeros([
                num_batch * batch_size * z_dim] - tf.shape(flatten),  # TODO
                                    dtype=flatten.dtype)
        padded = tf.concat([flatten, zero_padding], 0)
        embeddings = tf.reshape(padded, [num_batch, batch_size,
                                         embeddings.shape[1].value])
        return embeddings

    def _one_hot(self, indices, dim_vocab):
        return tf.one_hot(indices=indices, depth=dim_vocab)

    def _one_hot_padding(self, batch, max_length):
        padded_batch = []
        for one_hot in batch:
            vector = tf.cast(tf.reshape(one_hot, [-1]), tf.int32)
            zero_padding = tf.zeros([
                max_length * one_hot.shape[1].value] - tf.shape(vector),
                                    dtype=vector.dtype)
            padded = tf.concat([vector, zero_padding], 0)
            one_hot = tf.reshape(padded, [max_length, one_hot.shape[1].value])
            padded_batch.append(tf.cast(one_hot, dtype=vector.dtype))
        return tf.stack(padded_batch, axis=0)


if __name__ == '__main__':
    queue_reader = QueueReader("/Users/ousmane/Downloads/datasets/test/")
    with tf.train.MonitoredSession() as session:
        # session = tf.train.MonitoredSession()
        session.graph._unsafe_unfinalize()
        session.run(tf.global_variables_initializer())
        print("Embeddings", session.run(queue_reader.dequeue_many(5)['inputs']))

        while not session.should_stop():
            train_batch = queue_reader.dequeue_many(5)
            train = session.run(train_batch)
            print(train['inputs'])
