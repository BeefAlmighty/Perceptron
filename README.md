# Perceptron
Implementation of Perceptron algorithm with Visualization of learning process. 

# Sample Usage 

Designed for 2-dimensional binary classification with examples displaying in a [-10,10]x[-10,10]grid.  Perceptron initializes with 3-vector weights.


## Basic Use case: 

This is a fully balanced, linearly separable dataset with multivariate Gaussian distributed examples from each class.  

    n_samples=100
    pos = np.random.multivariate_normal([-5,1],np.eye(2),n_samples)
    neg = np.random.multivariate_normal([1,0],np.eye(2),n_samples)
    pos_examples=[(x[0], x[1], 1) for x in pos]
    neg_examples=[(x[0], x[1], 0) for x in neg]
    examples=pos_examples+neg_examples
    random.shuffle(examples)

    p=Perceptron([0,0,0])
    p.train(examples, steps=200)
