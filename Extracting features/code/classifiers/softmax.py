"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np
import math
from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    
      
    for i in range(X.shape[0]):
        ynew = np.matmul(X[i,:],W)
        y_exp = np.exp(ynew - np.max(ynew))  
        softmax_prob = y_exp/np.sum(y_exp)
        loss -= np.sum(np.log(softmax_prob[y[i]]))
        
        softmax_prob[y[i]] -= 1
        
        dW += np.matmul(X[i,:].reshape(X.shape[1],1),softmax_prob.reshape(1,W.shape[1]))
        
        
    loss +=  0.5 * reg * np.sum(np.square(W))
    loss /= X.shape[0]

    dW /= X.shape[0]
    dW += reg*W
   
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    ynew = np.matmul(X,W)
    y_exp = np.exp(ynew - np.max(ynew,axis=1,keepdims = True))
    softmax_prob = y_exp/np.sum(y_exp,axis=1,keepdims = True)
    all_Examples = range(X.shape[0])
    loss-= np.sum(np.log(softmax_prob[all_Examples,y]))
    
    softmax_prob[all_Examples,y] -= 1
    
    #softmax_prob/= X.shape[0]
    dW += np.matmul(np.transpose(X),softmax_prob)
    
    loss += 0.5 * reg * np.sum(np.square(W))
    loss /= X.shape[0]

    dW /= X.shape[0]
    dW += reg*W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 2e-7, 3e-7]
    regularization_strengths = [1e4, 2e4, 3e4, 4e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    for learn in learning_rates:
        for reg in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=learn, reg=reg,
                          num_iters=1500, verbose=True)
            y_pred_train = softmax.predict(X_train)
            train_acc = np.mean(y_train == y_pred_train)
            y_pred_val = softmax.predict(X_val)
            val_acc = np.mean(y_val == y_pred_val)
            results[(learn,reg)] = (train_acc,val_acc)
            all_classifiers.append(softmax)
            if(val_acc > best_val):
                best_val = val_acc
                best_softmax = softmax
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
