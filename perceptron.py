from scipy.stats import multivariate_normal
import random
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

class Perceptron():
    
    def __init__(self, weight=[0,0,0]):
        #type weight: list

        self.weight=np.array(weight)

    def predict(self, input):
        #type input: list

        input=np.array([1]+ input)
        temp=np.dot(self.weight, input)
        return int(temp>0)
    
    def train(self, examples, steps=60):
        '''
        type examples: list
        type steps: int

        train examples are labelled, (x,y,label in (0,1))
        steps is how many times we do stochastic training
        '''

        pos=np.array([[x[0],x[1]] for x in examples if x[-1]==1])
        neg=np.array([[x[0],x[1]] for x in examples if x[-1]==0])
        
        self.view(pos, neg)
        update=False

        for step in range(steps):
            #print('weight at step ',step, self.weight )
            example=random.choice(examples)
            label=example[-1]
            pred=self.predict([example[0], example[1]])
            if pred<label:
                self.weight=self.weight+[1, example[0], example[1]]
                update=True
            elif label< pred:
                self.weight=self.weight-[1, example[0], example[1]]
                update=True
           # self.view(pos, neg)
            if update: 
                self.view(pos, neg)
            update=False
    
    def view(self, pos, neg):
        '''
        type pos: numpy array of positive examples
        type neg: numpy array of negative examples
        '''

        def h(x,y):
            return self.predict([x,y])

        h=np.vectorize(h)
        x = np.linspace(-10, 10, 500)
        y = np.linspace(-10, 10, 500)
        X, Y = np.meshgrid(x, y)
        Z=h(X,Y)

        plt.contourf(X,Y,Z)
        plt.colorbar()
        plt.plot(pos[:,0], pos[:,1], 'gx')
        plt.plot(neg[:,0], neg[:,1], 'ro')
        plt.show()
