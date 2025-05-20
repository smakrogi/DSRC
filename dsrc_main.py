# Deep Sparse Representation-based Classification
"""
Deep Sparse Representation-based Classification (DSRC)
=====================================================
This module implements the Deep Sparse Representation-based Classification (DSRC) method as described in:
    M. Abavisani and V. M. Patel, "Deep sparse representation-based classification,"
    IEEE Signal Processing Letters, vol. 26, no. 6, pp. 948-952, June 2019.
    DOI:10.1109/LSP.2019.2913022
The implementation is based on TensorFlow 1.x and is built upon:
    - https://github.com/panji1990/Deep-subspace-clustering-networks
    - https://github.com/mahdiabavisani/Deep-multimodal-subspace-clustering-networks
Classes:
--------
ConvAE:
    Convolutional Autoencoder for deep sparse representation-based classification.
    - __init__: Initializes the model, builds the computation graph, and sets up training operations.
    - _initialize_weights: Initializes and stores model weights in a dictionary.
    - encoder: Builds the encoder part of the autoencoder.
    - decoder: Builds the decoder part of the autoencoder.
    - partial_fit: Performs a single optimization step for the main model.
    - pretrain_step: Performs a single optimization step for the pretraining stage.
    - initlization: Initializes model variables.
    - reconstruct: Reconstructs input data using the trained autoencoder.
    - transform: Extracts autoencoder features for input data.
    - save_model: Saves the model to disk.
    - restore: Restores the model from disk.
Functions:
----------
thrC(C, ro=0.1):
    Thresholds the coefficient matrix C by keeping the largest coefficients that sum up to a fraction 'ro' of the total L1 norm.
err_rate(gt_s, s):
    Computes the misclassification rate between ground truth labels and predicted labels.
testing(Img_test, Img_train, train_labels, test_labels, CAE, num_class, args):
    Trains and evaluates the ConvAE model on the provided training and testing data.
    Returns the final accuracy and coefficient matrix.
get_train_test_data(data, training_rate=0.8):
    Splits the dataset into training and testing sets based on the specified training rate.
    Returns training images, testing images, training labels, testing labels, and all labels.
Usage:
------
Run the script as a standalone program to train and evaluate DSRC on a dataset in .mat format.
Command-line arguments allow customization of dataset path, model name, training rate, number of epochs, and reporting frequency.
Dependencies:
-------------
- TensorFlow 1.x (compat.v1)
- NumPy
- SciPy
- argparse
- random
Author:
-------
Mahdi Abavisani
mahdi.abavisani@rutgers.edu
"""
# https://arxiv.org/abs/1904.11093
# Mahdi Abavisani
# mahdi.abavisani@rutgers.edu
# Built upon https://github.com/panji1990/Deep-subspace-clustering-networks
#        and https://github.com/mahdiabavisani/Deep-multimodal-subspace-clustering-networks
#
# Citation:  M. Abavisani and V. M. Patel, "Deep sparse representation-based clas- sification,"
#            IEEE Signal Processing Letters, vol. 26, no. 6, pp. 948-952, June 2019.
#            DOI:10.1109/LSP.2019.2913022

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as sio
import argparse
import random
tf.disable_v2_behavior()


class ConvAE(object):
    """
    ConvAE is a Convolutional Autoencoder model for unsupervised learning and self-expressive subspace clustering.
    Attributes:
        n_input (tuple): Shape of the input data (height, width).
        kernel_size (list): List of kernel sizes for each convolutional layer.
        n_hidden (list): List of number of filters for each convolutional layer.
        batch_size (int): Total batch size (train + test).
        train_size (int): Number of training samples in each batch.
        test_size (int): Number of test samples in each batch (batch_size - train_size).
        reg (float or None): Regularization parameter (optional).
        model_path (str or None): Path to save the trained model.
        restore_path (str or None): Path to restore a saved model.
        iter (int): Training iteration counter.
        sess (tf.InteractiveSession): TensorFlow session for running computations.
        saver (tf.train.Saver): TensorFlow Saver object for saving/restoring models.
        summary_writer (tf.summary.FileWriter): Writer for TensorBoard summaries.
    Methods:
        __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, batch_size=200, train_size=100, reg=None, denoise=False, model_path=None, restore_path=None, logs_path='./logs'):
            Initializes the ConvAE model, builds the computation graph, and sets up training operations.
        _initialize_weights(self):
            Initializes and returns a dictionary of model weights and biases.
        encoder(self, X, weights):
            Builds the encoder part of the autoencoder.
            Args:
                X (tf.Tensor): Input tensor.
                weights (dict): Dictionary of weights and biases.
            Returns:
                latent (tf.Tensor): Latent representation.
                latents (tf.Tensor): Output of the last encoder layer before the final latent layer.
                shapes (list): List of shapes for each encoder layer.
        decoder(self, z, weights, shapes):
            Builds the decoder part of the autoencoder.
            Args:
                z (tf.Tensor): Latent representation.
                weights (dict): Dictionary of weights and biases.
                shapes (list): List of shapes for each encoder layer.
            Returns:
                recons (tf.Tensor): Reconstructed input.
        partial_fit(self, X, Y, lr):
            Performs a single optimization step on the main loss.
            Args:
                X (np.ndarray): Test data batch.
                Y (np.ndarray): Training data batch.
                lr (float): Learning rate.
            Returns:
                cost (float): Reconstruction loss.
                Coef (np.ndarray): Self-expressive coefficient matrix.
        pretrain_step(self, X, Y, lr):
            Performs a single optimization step on the pretraining loss.
            Args:
                X (np.ndarray): Test data batch.
                Y (np.ndarray): Training data batch.
                lr (float): Learning rate.
            Returns:
                cost (float): Pretraining reconstruction loss.
        initlization(self):
            Initializes all TensorFlow variables.
        reconstruct(self, X):
            Reconstructs the input data using the trained autoencoder.
            Args:
                X (np.ndarray): Input data.
            Returns:
                np.ndarray: Reconstructed data.
        transform(self, X, Y):
            Computes the autoencoder features for the given data.
            Args:
                X (np.ndarray): Test data batch.
                Y (np.ndarray): Training data batch.
            Returns:
                np.ndarray: Autoencoder features.
        save_model(self):
            Saves the current model to the specified model_path.
        restore(self):
            Restores the model from the specified restore_path.
    """

    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, batch_size=200, train_size=100,reg=None, \
                 denoise=False, model_path=None, restore_path=None, \
                 logs_path='./logs'):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = batch_size - train_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        
        tf.set_random_seed(2019)
        weights = self._initialize_weights()

        # input required to be fed
        
        self.train = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
        self.test = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [],name='learningRate')

        self.x = tf.concat([self.train, self.test], axis=0) #Concat testing and training samples

        # Encoder
        latent, latents, shape = self.encoder(self.x, weights)
        latent_shape = tf.shape(latent)

        # Slice the latent space features to separate training and testing latent features
        latent_train =  tf.slice(latent,[0,0,0,0],[self.train_size, latent_shape[1], latent_shape[2], latent_shape[3]])
        latent_test =  tf.slice(latent,[self.train_size,0,0,0],[self.test_size, latent_shape[1], latent_shape[2], latent_shape[3]])

        # Vectorize the features
        z_train = tf.reshape(latent_train, [self.train_size, -1])
        z_test = tf.reshape(latent_test, [self.test_size, -1])
        z = tf.reshape(latent, [self.batch_size, -1])

        # Self-expressive layer
        # Coef is the self-expressive coefficient matrix
        Coef = weights['Coef']   # This is \theta in the paper

        # Approximate the latent features of the test samples using the training samples
        # sparsity is enforced by the self-expressive layer.
        z_test_c = tf.matmul(Coef, z_train)
        z_c = tf.concat([z_train, z_test_c], axis=0)
        latent_c_test = tf.reshape(z_test_c, tf.shape(latent_test)) 
          
        latent_c_pretrain =  tf.concat([latent_train, latent_test], axis=0) # used in pretraining stage
        latent_c =  tf.concat([latent_train, latent_c_test], axis=0)        # used in the main model

        self.x_r_pretrain = self.decoder(latent_c_pretrain, weights,  shape) # used in pretraining stage
        self.x_r = self.decoder(latent_c, weights,  shape)                   # used in the main model            


        self.Coef_test = Coef
        
        self.AE =  tf.concat([z_train, z_test], axis=0) # Autoencoder features to be used in benchmarks comparison


        # l_2 reconstruction loss

        self.loss_pretrain = tf.reduce_sum(tf.pow(tf.subtract(self.x, self.x_r_pretrain), 2.0))
        
        self.reconst_cost_x = tf.reduce_sum(tf.pow(tf.subtract(self.x, self.x_r), 2.0))
        tf.summary.scalar("recons_loss", self.reconst_cost_x)
        
        # Regularization term
        self.reg_losses = tf.reduce_sum(tf.pow(Coef, 2.0))
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        # Self-expressive loss
        # The self-expressive loss is the l_2 norm of the difference between the latent features of the test samples
        # and the approximate latent features of the test samples
        # using the training samples
        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0))

        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses)

        # TOTAL LOSS
        self.loss = self.reconst_cost_x + reg_constant1 * self.reg_losses + 0.5 * re_constant2 * self.selfexpress_losses

        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss)  # GradientDescentOptimizer #AdamOptimizer
        self.optimizer_pretrain = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss_pretrain)  # GradientDescentOptimizer #AdamOptimizer

        self.init = tf.global_variables_initializer()
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=tfconfig)
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]) # to save the pretrained model
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


    def _initialize_weights(self):
        '''
        initializes weights for the model and stores them in a dictionary.
        '''
        
        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0",
                                                shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                        self.n_hidden[0]],
                                                initializer=tf.compat.v1.glorot_normal_initializer                                                            )
        all_weights['enc1_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['enc_w1'] = tf.get_variable("enc_w1",
                                                shape=[self.kernel_size[1], self.kernel_size[1],
                                                        self.n_hidden[0],
                                                        self.n_hidden[1]],
                                                initializer=tf.compat.v1.glorot_normal_initializer)
        all_weights['enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

        all_weights['enc_w2'] = tf.get_variable("enc_w2",
                                                shape=[self.kernel_size[2], self.kernel_size[2],
                                                        self.n_hidden[1],
                                                        self.n_hidden[2]],
                                                initializer=tf.compat.v1.glorot_normal_initializer)
        all_weights['enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype=tf.float32))

        all_weights['dec_w0'] = tf.get_variable("dec1_w0",
                                                shape=[self.kernel_size[2], self.kernel_size[2],
                                                        self.n_hidden[1],
                                                        self.n_hidden[3]],
                                                initializer=tf.compat.v1.glorot_normal_initializer)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

        all_weights['dec_w1'] = tf.get_variable("dec1_w1",
                                                shape=[self.kernel_size[1], self.kernel_size[1],
                                                        self.n_hidden[0],
                                                        self.n_hidden[1]],
                                                initializer=tf.compat.v1.glorot_normal_initializer)
        all_weights['dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['dec_w2'] = tf.get_variable("dec1_w2",
                                                shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                        self.n_hidden[0]],
                                                initializer=tf.compat.v1.glorot_normal_initializer)
        all_weights['dec_b2'] = tf.Variable(tf.zeros([1], dtype=tf.float32))

        all_weights['enc_w3'] = tf.get_variable("enc_w3",
                                                shape=[self.kernel_size[3], self.kernel_size[3],
                                                       self.n_hidden[2],
                                                       self.n_hidden[3]],
                                                initializer=tf.compat.v1.glorot_normal_initializer)
        all_weights['enc_b3'] = tf.Variable(tf.zeros([self.n_hidden[3]], dtype=tf.float32))

        all_weights['Coef'] = tf.Variable(1.0e-4 * tf.ones([self.test_size, self.train_size], tf.float32), name='Coef')

        return all_weights


    # Building the encoder
    def encoder(self, X, weights):
        """
        Encodes the input tensor X using a series of convolutional layers with ReLU activations.
        Args:
            X (tf.Tensor): Input tensor of shape [batch_size, height, width, channels].
            weights (dict): Dictionary containing the weights and biases for each convolutional layer.
                Expected keys:
                    - 'enc_w0', 'enc_b0': Weights and biases for the first conv layer.
                    - 'enc_w1', 'enc_b1': Weights and biases for the second conv layer.
                    - 'enc_w2', 'enc_b2': Weights and biases for the third conv layer.
                    - 'enc_w3': Weights for the final conv layer.
        Returns:
            tuple:
                - latent (tf.Tensor): Output tensor after the final convolution and ReLU activation.
                - latents (tf.Tensor): Output tensor from the third convolutional layer (before the final conv).
                - shapes (list): List of shapes of the tensors at each stage of the encoder.
        """
        
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(X.get_shape().as_list())

        layer1 = tf.nn.bias_add(
            tf.nn.conv2d(X, weights['enc_w0'], strides=[1, 2, 2, 1], padding='SAME'),
            weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        layer2 = tf.nn.bias_add(
            tf.nn.conv2d(layer1, weights['enc_w1'], strides=[1, 1, 1, 1], padding='SAME'),
            weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)
        layer3 = tf.nn.bias_add(
            tf.nn.conv2d(layer2, weights['enc_w2'], strides=[1, 2, 2, 1], padding='SAME'),
            weights['enc_b2'])
        layer3 = tf.nn.relu(layer3)
        latents = layer3
        print(layer3.shape)
 
        shapes.append(layer1.get_shape().as_list())
        shapes.append(layer2.get_shape().as_list())
        layer3_in = layer3

        latent = tf.nn.conv2d(layer3_in, weights['enc_w3'], strides=[1, 1, 1, 1], padding='SAME')
        latent = tf.nn.relu(latent)
        shapes.append(latent.get_shape().as_list())

        return latent, latents, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        """
        Decodes the latent representation `z` into a reconstructed output using a series of transposed convolutional layers.
        Args:
            z (tf.Tensor): Latent representation tensor to be decoded.
            weights (dict): Dictionary containing the decoder weights and biases with keys 'dec_w0', 'dec_b0', 'dec_w1', 'dec_b1', 'dec_w2', 'dec_b2'.
            shapes (list): List of shapes for each decoder layer output, where each shape is a list or tuple specifying the output dimensions.
        Returns:
            tf.Tensor: The reconstructed output tensor after passing through the decoder network.
        """

        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack(
            [tf.shape(self.x)[0], shape_de1[1], shape_de1[2], shape_de1[3]]), \
                                               strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['dec_w1'], tf.stack(
            [tf.shape(self.x)[0], shape_de2[1], shape_de2[2], shape_de2[3]]), \
                                               strides=[1, 1, 1, 1], padding='SAME'), weights['dec_b1'])
        layer2 = tf.nn.relu(layer2)
        shape_de3 = shapes[0]
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['dec_w2'], tf.stack(
            [tf.shape(self.x)[0], shape_de3[1], shape_de3[2], shape_de3[3]]), \
                                               strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b2'])
        layer3 = tf.nn.relu(layer3)
        recons = layer3

        return recons

    def partial_fit(self, X,Y, lr):
        """
        Performs a single partial fit (training step) on the model using the provided input and target data.
        Args:
            X (numpy.ndarray or compatible): Input data for testing or inference.
            Y (numpy.ndarray or compatible): Target data for training.
            lr (float): Learning rate for the optimizer.
        Returns:
            tuple: A tuple containing:
                - cost (float): The reconstruction cost after the training step.
                - Coef (numpy.ndarray): The coefficient matrix obtained after the training step.
        Side Effects:
            - Updates the model's internal iteration counter (`self.iter`).
            - Writes summary data to the summary writer for visualization/logging.
        """

        cost, summary, _, Coef = self.sess.run(
            (self.reconst_cost_x, self.merged_summary_op, self.optimizer, self.Coef_test), feed_dict={self.learning_rate:lr,self.train:Y,self.test:X})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef
    
    def pretrain_step(self, X,Y, lr):
        def pretrain_step(self, X, Y, lr):
            """
            Performs a single pretraining step for the model.
            Executes a TensorFlow session run to optimize the pretraining objective using the provided input and target data,
            updates the summary writer for TensorBoard visualization, and increments the training iteration counter.
            Args:
                X: Input data for the pretraining step (typically used as test data).
                Y: Target data for the pretraining step (typically used as training data).
                lr: Learning rate for the optimizer.
            Returns:
                cost: The reconstruction cost computed during this pretraining step.
            """

        cost, summary, _ = self.sess.run(
            (self.reconst_cost_x, self.merged_summary_op, self.optimizer_pretrain), feed_dict={self.learning_rate:lr,self.train:Y,self.test:X})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x:X})

    def transform(self,  X,Y):
        return self.sess.run(self.AE, feed_dict={self.train:Y,self.test:X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")



def thrC(C, ro=0.1):
    """
    Thresholds the columns of matrix C based on the cumulative sum of absolute values.
    For each column in C, retains the largest elements such that their cumulative absolute sum
    exceeds a fraction `ro` of the total absolute sum of the column. The remaining elements are set to zero.
    If `ro >= 1`, the original matrix C is returned unchanged.
    Parameters
    ----------
    C : np.ndarray
        Input 2D array (matrix) to be thresholded.
    ro : float, optional
        Fraction of the total absolute sum to retain in each column (default is 0.1).
    Returns
    -------
    Cp : np.ndarray
        Thresholded matrix with the same shape as C, where only the largest elements in each column
        (by absolute value) are retained to satisfy the cumulative sum condition.
    """

    if ro < 1:
        N1 = C.shape[0]
        N2 = C.shape[1]
        Cp = np.zeros((N1, N2))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N2):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def err_rate(gt_s, s):
    """
    Calculates the error rate (miss rate) between two sequences.
    Parameters:
        gt_s (np.ndarray): Ground truth sequence (1D array).
        s (np.ndarray): Predicted or estimated sequence (1D array) to compare against the ground truth.
    Returns:
        float: The proportion of elements in which the two sequences differ (error rate).
    Notes:
        - Both input arrays must have the same shape.
        - The function computes the number of mismatches and divides by the total number of elements.
    """

    err_x = np.sum(gt_s[:] != s[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate



def testing(Img_test,Img_train, train_labels,test_labels, CAE, num_class,args):
    """
    Performs testing and training steps for a Convolutional Autoencoder (CAE) based classification model.
    Args:
        Img_test (array-like): Test images.
        Img_train (array-like): Training images.
        train_labels (array-like): Labels for training images.
        test_labels (array-like): Labels for test images.
        CAE (object): Convolutional Autoencoder model instance with required methods.
        num_class (int): Number of classes in the dataset.
        args (argparse.Namespace): Arguments containing 'max_step', 'pretrain_step', and 'display_step'.
    Returns:
        tuple:
            acc_x (float): Final accuracy achieved on the test set.
            Coef (numpy.ndarray): Final coefficient matrix from the CAE model.
    Notes:
        - The function performs a pretraining phase followed by a main training phase.
        - During training, it periodically prints cost and accuracy.
        - Optionally, it can save accuracy, coefficients, and cost to a .mat file.
    """

    Img_test = np.array(Img_test)
    Img_test = Img_test.astype(float)
    Img_train = np.array(Img_train)
    Img_train = Img_train.astype(float)

    train_labels = np.array(train_labels[:])
    train_labels = train_labels - train_labels.min() + 1
    train_labels = np.squeeze(train_labels)

    test_labels = np.array(test_labels[:])
    test_labels = test_labels - test_labels.min() + 1
    test_labels = np.squeeze(test_labels)

    CAE.initlization()
    max_step = args.max_step  # 500 + num_class*25# 100+num_class*20
    pretrain_max_step = args.pretrain_step
    display_step = args.display_step #max_step
    lr = 1.0e-3
    
    
    epoch = 0
    class_ = np.zeros(np.max(test_labels))
    prediction = np.zeros(len(test_labels))
    ACC =[]
    Cost=[]
    
    while epoch < pretrain_max_step:
        epoch = epoch + 1
        cost = CAE.pretrain_step(Img_test,Img_train, lr)  #

        if epoch % display_step == 0:
            print ("pretrain epoch: %.1d" % epoch, "cost: %.8f" % (cost / float(batch_size)))   
    

    while epoch < max_step:
        epoch = epoch + 1
        cost, Coef = CAE.partial_fit(Img_test,Img_train, lr)  #

        if epoch % display_step == 0:
            print ("epoch: %.1d" % epoch, "cost: %.8f" % (cost / float(batch_size)))   
            Coef = thrC(Coef)
            Coef= np.abs(Coef)
            for test_sample in range(0,len(test_labels)):
                x = Coef[test_sample,:]
                for l in range(1,np.max(test_labels)+1):
                    l_idx = np.array([j for j in range(0,len(train_labels)) if train_labels[j]==l])
                    l_idx= l_idx.astype(int)
                    class_[int(l-1)] = sum(np.abs(x[l_idx]))
                prediction[test_sample] = np.argmax(class_) +1

            prediction = np.array(prediction)
            missrate_x = err_rate(test_labels, prediction)
            acc_x = 1 - missrate_x
            print("accuracy: %.4f" % acc_x)
            ACC.append(acc_x)
            Cost.append(cost / float(batch_size))
    if False: # change to ture to save values in a mat file
        sio.savemat('./coef.mat', dict(ACC=ACC,Coef=Coef,Cost=Cost))

    return acc_x, Coef

def get_train_test_data(data,training_rate=0.8):
    '''
    Extracts features and labels from the dictionary "data," and splits the samples
    into training and testing sets.
    
    Input:
        data: dictionary containing two keys: {feature, Label}
            data['features'] : vectorized features (1024 x N)
            data['Label']   : groundtruth labels (1 x N)
        rate: ratio of the # of training samples to the total # of samples
        
    Output:
        training and testing sets.
            
    '''

    Label = data['Label']
    Label = np.squeeze(np.array(Label))
    training_size = int(training_rate * len(Label))

    perm = np.random.permutation(len(Label))
    training_idx = perm[:training_size]
    testing_idx = perm[training_size:]

    train_labels = Label[training_idx]
    test_labels = Label[testing_idx]


    I_test = []
    I_train = []
    img = data['features']
    training_img = img[:,training_idx]
    testing_img = img[:,testing_idx]

    for i in range(training_img.shape[1]):
        temp = np.reshape(training_img[:, i], [32, 32])
        I_train.append(temp)
    Img_train = np.transpose(np.array(I_train), [0, 2, 1])
    Img_train = np.expand_dims(Img_train[:], 3)

    for i in range(testing_img.shape[1]):
        temp = np.reshape(testing_img[:, i], [32, 32])
        I_test.append(temp)
    Img_test = np.transpose(np.array(I_test), [0, 2, 1])
    Img_test = np.expand_dims(Img_test[:], 3)

    return Img_train,Img_test,train_labels,test_labels,Label

if __name__ == '__main__':
    
    random.seed(2019)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mat', dest='mat', default='umd', help='path of the dataset')
    parser.add_argument('--model', dest='model', default='umd',
                        help='name of the model to be saved')
    parser.add_argument('--rate', dest='rate', type=float, default=0.8, help='Pecentage of samples ')
    parser.add_argument('--epoch', dest='max_step', type=int, default=10000, help='Max # training epochs')
    parser.add_argument('--pretrain_step', dest='pretrain_step', type=int, default=1000, help='Max # of pretraining epochs ')
    parser.add_argument('--display_step', dest='display_step', type=int, default=1000, help='frequency of reports')


    args = parser.parse_args()



    # load face images and labels
    datapath = './data/' + args.mat + '.mat'
    data = sio.loadmat(datapath) 


    # Split the data into training and testing sets
    [Im_train,Im_test,train_labels,test_labels,Label] = get_train_test_data(data,training_rate=args.rate)


    
    # face image clustering
    n_input = [32, 32]
    kernel_size = [5,3,3,1]
    n_hidden = [10, 20, 30,30]

    iter_loop = 0
    
    num_class = Label.max()
    batch_size = len(Label)
    training_size = len(train_labels)

    # These regularization values work best if the features are intensity values between 0-225
    reg1 = 1.0  # random.uniform(1, 10)
    reg2 = 8.0 # random.uniform(1, 10)

    model_path = './models/' + args.model + '.ckpt'
    logs_path = './logs'
    tf.reset_default_graph()
    CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, \
                 kernel_size=kernel_size, batch_size=batch_size, train_size=training_size,model_path=model_path, restore_path=model_path,
                 logs_path=logs_path)

    ACC, C = testing(Im_test,Im_train, train_labels, test_labels, CAE, num_class,args)



























