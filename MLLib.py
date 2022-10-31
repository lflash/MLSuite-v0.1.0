# %%
# (0.0) Imports, inits and TODOs
# TODO: next major version will have:
        # - python logging instead of printing/verbosity
        # - __init__ funcs which use the super.__init__() function so args/optional args are explicit
        # - unit testing as it's being built
# TODO: calculate variance in Δparam in each batch. split largest variance node into 2 nodes, one with the positive applied, one with negative
# TODO: wrapper module
# TODO: convolution module
# TODO: downsample/pooling module
# TODO: learning rate investigator function
    # as network size increases
    # test time to converge, max score achieved
# TODO: GAN flush function should run it til it converges-if it converges
# TODO: add option in backprop to randomise lowest contributing/wrongly contributing nodes
# TODO: Tokeniser/transformer classes
# TODO: turn tests in cells into functions
# TODO: try make I/O = Z rather than A
# TODO: sinusoidal network
# TODO: stable diffusion network

import numpy as np
import random
import json
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
from scipy import signal

# warnings.simplefilter('error')
np.set_printoptions(linewidth = 1000, suppress=True)

def inc(x):
    # print('inc got ',x)
    x[0] = x[0]+1
    # print('returning ',x)
    return x[0]

def normalise(X):
    retval = (X-np.sum(X))          # centre on 0
    retval = retval/np.max(retval)  # 
    return retval

def numToArray(num):
    retArray = np.zeros(10)
    retArray[num] = 1
    return retArray

def sigmoid(z):
    return (1.0/(1.0+np.exp(-z)))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def swish(z):
    return np.nan_to_num(z*sigmoid(z))

def swish_prime(z):
    return np.nan_to_num((np.exp(z))*(z + np.exp(-z) + 1)/((np.exp(z)+1)**2))

def relu(z):    
    return np.vectorize(lambda x: float(x>0)*x)(z)

def relu_prime(z):
    return np.vectorize(lambda x: float(x>0))(z)

def linear(z):
    return z

def linear_prime(z):
    return np.vectorize(lambda x: 1)(z)

def sins(z):
    k=[-1]
    #print(z)
    try:
        N=len(z)
    except:
        N=1
    #print('sins: ', np.vectorize(lambda x: np.cos(2*np.pi*inc(k)*x/N))(z))
    # return np.fft.rfft(z)
    return np.vectorize(lambda x: np.cos(2*np.pi*inc(k)*x/N))(z)
    
def sins_prime(z):
    k1=[-1]
    k2=[-1]
    N=len(z)
    #print('sins_prime: ', np.vectorize(lambda x: -2*np.pi*inc(k1)*np.sin(2*np.pi*inc(k2)*x/N)/N)(z))
    # return np.fft.irfft(z)
    return np.vectorize(lambda x: -2*np.pi*inc(k1)*np.sin(2*np.pi*inc(k2)*x/N)/N)(z)

def zero_pad(x, padDims, padToShape = False, corner = 'tl'):
    (xx,xy) = x.shape
    (px,py) = padDims
    if padToShape:
        retval = np.zeros((px,py))
    else:
        retval = np.zeros((xx+px,xy+py))
    match corner:
        case 'tl':
            stampLoc = (slice(0,xx),slice(0,xy))
        case 'tr':
            stampLoc = (slice(0,xx),slice(-xy,None))
        case 'bl':
            stampLoc = (slice(-xx,None),slice(0,xy))
        case 'br':
            stampLoc = (slice(-xx,None),slice(-xy,None))
   
    retval[stampLoc[0],stampLoc[1]] = x

    return retval

def zero_strip(x, stripDims, stripToShape = False, corner = 'tl'):
    (xx,xy) = x.shape
    (px,py) = stripDims
    if stripToShape:
        (sx,sy) = (px,py)
    else:
        (sx,sy) = (xx-px,xy-py)
    
    match corner:
        case 'tl':
            retval = x[slice(0,sx),slice(0,sy)]
        case 'tr':
            retval = x[slice(0,sx),slice(-sy,None)]
        case 'bl':
            retval = x[slice(-sx,None),slice(0,sy)]
        case 'br':
            retval = x[slice(-sx,None),slice(-sy,None)]
   
    return retval

def conv(a, b, mode = 'valid'): 
    # Returns function c such that a * b = c.
    # outputDims = [(a.shape[0]-b.shape[0]), (a.shape[1]-b.shape[1])]
    
    match mode:
        case 'full':
            aPad = tuple(d-2 for d in b.shape)
            a = zero_pad(zero_pad(a,aPad,corner='tl'),aPad,corner='tl')
            ashape = np.shape(a)
            zb = zero_pad(b,ashape,padToShape=True)

        case 'valid':
            ashape = np.shape(a)
            zb = zero_pad(b,ashape,padToShape=True)
    # print('a.shape: ', a.shape)
    # print('b.shape: ', b.shape)
    # print('(a.shape[0]-b.shape[0]): ', (a.shape[0]-b.shape[0]))
    # print('(a.shape[1]-b.shape[1]): ', (a.shape[1]-b.shape[1]))
    # Convert larger polynomial using fft

    ffta = np.fft.fftn(a)
    

    fftb = np.fft.fftn(zb)
    #fftb = np.fft.fftn(b,ashape)
    
    # Divide the two in frequency domain

    fftquotient = ffta * fftb
    
    # Convert back to polynomial coefficients using ifft
    # Should give c but with some small extra components

    c = np.fft.ifftn(fftquotient)
    
    # Get rid of imaginary part and round up to 6 decimals
    # to get rid of small real components

    trimmedc = np.around(np.real(c),decimals=8)
    
    # Trim zeros and eliminate
    # empty rows of coefficients
    
    # cleanc = trim_zero_empty(trimmedc)
                
    # return cleanc

    # print('a.shape: ', a.shape)
    # print('b.shape: ', b.shape)
    # print('(a.shape[0]-b.shape[0]): ', (a.shape[0]-b.shape[0]))
    # print('(a.shape[1]-b.shape[1]): ', (a.shape[1]-b.shape[1]))

    ####return trimmedc[b.shape[0]:, b.shape[1]:]
    match mode:
        case 'full':
            # aPad = tuple(d-2 for d in b.shape)
            # a = zero_pad(zero_pad(a,aPad,corner='tl'),aPad,corner='tl')
            return trimmedc

        case 'valid':
            return zero_strip(trimmedc,tuple(d-1 for d in b.shape),corner='br')

def iconv(a, b, mode = 'valid'):   
    # Returns function c such that a = b * c.
    
    validOutDim = lambda x1,x2: x2 + (x1-1)*(x1>x2) + (-x1+1)*(x1<x2)
    # outShape = tuple(validOutDim(d1,d2) for d1,d2 in zip(a.shape,b.shape))
    # maxShape = tuple(max(d1,d2,d3) for d1,d2,d3 in zip(outShape,a.shape,b.shape))
    match mode:
        case 'full':
            # aPad = tuple(d-2 for d in b.shape)
            # a = zero_pad(zero_pad(a,aPad,corner='tl'),aPad,corner='tl')
            outShape = tuple(validOutDim(d1,d2) for d1,d2 in zip(a.shape,b.shape))
            maxShape = tuple(max(d1,d2,d3) for d1,d2,d3 in zip(outShape,a.shape,b.shape))
            a = zero_pad(a,maxShape,padToShape=True,corner='br')
            ashape = np.shape(a)
            zb = zero_pad(b,ashape,padToShape=True)
            
        case 'valid':
            outShape = tuple(validOutDim(d1,d2) for d1,d2 in zip(a.shape,b.shape))
            maxShape = tuple(max(d1,d2,d3) for d1,d2,d3 in zip(outShape,a.shape,b.shape))
            a = zero_pad(a,maxShape,padToShape=True,corner='br')
            ashape = np.shape(a)
            zb = zero_pad(b,ashape,padToShape=True)
            pass


    #outShape = outDim(a,b)
    
    #print('outShape',outShape)
    # Convert larger polynomial using fft

    ffta = np.fft.fftn(a)
    
    
    fftb = np.fft.fftn(zb)
    # fftb = np.fft.fftn(b,ashape)
    
    # Divide the two in frequency domain

    fftquotient = ffta / fftb
    
    # Convert back to polynomial coefficients using ifft
    # Should give c but with some small extra components

    c = np.fft.ifftn(fftquotient)
    
    # Get rid of imaginary part and round up to 6 decimals
    # to get rid of small real components

    trimmedc = np.around(np.real(c),decimals=8)
    
    # Trim zeros and eliminate
    # empty rows of coefficients
    
    # cleanc = trim_zero_empty(trimmedc)
                
    # return cleanc
    match mode:
        case 'full':
            stripDims = tuple(d-1 for d in b.shape)
            return zero_strip(zero_strip(trimmedc,stripDims,corner='tl'),stripDims,corner='br')

        case 'valid':
            return zero_strip(trimmedc,outShape,stripToShape=True)
            
def get_training_data():

    (trainingImages, trainingDigits), (testImages, testDigits) = tf.keras.datasets.mnist.load_data()

    # trainingOutput = [numToArray(i) for i in trainingDigits]
    # testOutput = [numToArray(i) for i in testDigits]

    trainingImages = [(trainingImages[i]/255) for i in range(len(trainingDigits))]
    testImages = [(testImages[i]/255) for i in range(len(testDigits))]

    # trainingInputs = [(trainingImages[i]).flatten() for i in range(len(trainingDigits))]
    # testInputs = [(testImages[i]).flatten() for i in range(len(testDigits))]

    trainingData = [{'X':trainingImages[i],'Y':trainingDigits[i]} for i in range(len(trainingImages))]
    testData = [{'X':testImages[i],'Y':testDigits[i]} for i in range(len(testImages))]

    return (trainingData, testData)


# %%
# (1.0) MLModule Base Class

class MLModule():

    # init function for every MLModule child
    def __init__(self, fromStr = '', comment = '', verbose = False, **kwargs):
        
        if fromStr != '':
            self.fromStr(fromStr)
            
        else:
            
            self.zero(**kwargs)
            self.comment = comment
            self.verbose = verbose  # used to diagnose problems when testing modules
            self.learningRate = 1

    # will be overridden by all child classes
    def zero(self, **kwargs):
        
        self.verbose = False
        self.comment = ''


    # prepare a class variable for conversion to JSON
    def serialise(self, v):
        # print('in serialise: type: ', str(type(v)))
        match str(type(v)):
            case "<class 'numpy.ndarray'>":
                retval = 'np.array('+str(v).replace('\\n','\n')+')'#.replace(']\n [','],\n [')+')'
                # retval = 'np.array('+str(v.tolist())+')'
            case "<class 'slice'>":
                retval = str(v)
            case "<class 'list'>":
                retval = []
                for l in v:
                    retval = retval + [self.serialise(l)]
            case "<class 'function'>":
                retval = v.__name__

            case _:
                retval = v
        return retval
    
    # undo serialisation for JSON
    def deserialise(self, s):
        match str(type(s)):
            case "<class 'str'>":
                try:
                    retval = eval(s)
                except:
                    retval = s

            case "<class 'list'>":
                retval = [self.deserialise(l) for l in s]             

            case _:
                retval = s
        return retval

    # convert class to dictionary
    def toDict(self):
        retDict = {}
        
        for key in vars(self):
            retDict[key] = self.serialise(eval('self.'+key))

        return retDict

    # convert dictionary to class
    def fromDict(self, inputDict):
        
        for k in inputDict.keys():
            
            exec('self.' + k + ' = self.deserialise(inputDict[k])')
            
        return self
    
    # convert class to JSON format
    def JSON(self):
        return json.dumps(self.toDict(), indent=4, sort_keys=True)
    
    # convert JSON obj to class obj
    def fromJSON(self, JSONobj):
        return self.fromDict(json.loads(JSONobj))

    # returns legible string for printing
    def __str__(self):
        return str(self.JSON()).replace('\\n','\n')

    # converts str to class obj
    def fromStr(self, inputStr):
        return self.fromDict(json.loads(inputStr, strict=False))

    # to be overridden: push input, pop output
    def push(self, input):
        return self.output()

    # to be overridden: maintain structure, flush IO dependent vars + nablas with 0s
    def flush(self):
        return self

    # to be overridden: return output
    def output(self):
        return 0

    # # to be overridden: takes ideal output (feedback) for current state. uses it to learn
    def acceptFeedback(self, feedback):
        return self

    # to be overridden: module may need to meet initialiation criteria after flush
    def isInitialised(self):
        return True

    # so obj can be used as dict key
    def hash(self):
        return(hash(str(self)))

# %%
# (1.1) NN base class

class NN(MLModule):
    # def __init__(self, layerSizes, learningRate = 0.05, batchSize = 10, activation = sigmoid, activation_prime = sigmoid_prime):
    #     super.__init__(self, layerSizes, learningRate=learningRate, batchSize=batchSize, activation=activation, activation_prime=activation_prime)

    def zero(self, **kwargs):
        # print('kwargs: ', kwargs)

        crucialIncludes = ['layerSizes']
        for k in crucialIncludes:
            if k not in kwargs.keys():
                raise Exception(k + ' not included in keyed args. must include: ', crucialIncludes)
            else:
                exec('self.' + k + ' = '+str(kwargs[k]))


        # optional includes
        optionalIncludes = {'learningRate': 0.05, 'batchSize': 10, 'activation': 'sigmoid', 'activation_prime': 'sigmoid_prime'}
        for (k, v) in optionalIncludes.items():
            # print(k, v)
            if k not in kwargs.keys():
                # print('self.' + k + ' = '+str(v))
                exec('self.' + k + ' = '+str(v))
            else:
                exec('self.' + k + ' = '+str(kwargs[k]))
        
        self.feedbackCount = 0

        # A[l+1] = Γ(W[l].A[l] + B[l])

        self.A = [np.zeros((s, 1)) for s in self.layerSizes]
        
        self.B = [np.random.randn(s, 1) for s in self.layerSizes[1:]]
        self.W = [np.random.randn(self.layerSizes[i+1],self.layerSizes[i]) for i in range(len(self.layerSizes[:-1]))]    # TODO: Neurons should only connect to a small number of other neurons, not all. could fix w/ fimple filter func
                       
        self.nabla_W = [W*0 for W in self.W]
        self.nabla_B = [B*0 for B in self.B]

    def output(self):
        return self.A[-1]

    def feedForward(self):
        for l in range(len(self.A[:-1])):
            if self.verbose:
                print('l:', l)
                print('self.A[l+1].shape:', self.A[l+1].shape)
                print('self.W[l].shape:', self.W[l].shape)
                print('self.A[l].shape:', self.A[l].shape)
                print('self.B[l].shape:', self.B[l].shape)
            self.A[l+1] = self.activation(np.dot(self.W[l], self.A[l])+self.B[l])

        return self.output()
    
    def push(self, input):
        self.A[0] = np.reshape(input,self.A[0].shape)
        return self.feedForward()

    def cost_derivative(self, Y_ideal):
        Y_ideal = np.reshape(Y_ideal,self.A[-1].shape)
                
        return (self.A[-1]-Y_ideal)

    def acceptFeedback(self, feedback, applyChanges = True):
        
        δC_δAlp1 = self.cost_derivative(feedback)
        return self.acceptNabla(δC_δAlp1, applyChanges = applyChanges)
    
    def acceptNabla(self, δC_δAlp1, applyChanges = True):

        nabla_B = []
        nabla_W = []

        # backprop algorithm
        for l in list(range(len(self.B)))[::-1]:
                
            Zlp1 = np.dot(self.W[l], self.A[l]) + self.B[l]
            δAlp1_δZlp1 = self.activation_prime(Zlp1)
            δZlp1_δWl = self.A[l]
            δZlp1_δBl = 1



            if self.verbose:
                print('l: ', l)
                print('δC_δAlp1.shape: ',δC_δAlp1.shape)
                print('δAlp1_δZlp1.shape: ',δAlp1_δZlp1.shape)
                print('δZlp1_δWl.shape: ',δZlp1_δWl.shape)
                print('δC_δAlp1:    ',δC_δAlp1)
                print('δAlp1_δZlp1: ',δAlp1_δZlp1)
                print('δZlp1_δWl:   ',δZlp1_δWl)
                # print('δZlp1_δBl.shape: ',δZl_δBl.shape)
                print()
            
            nabla_B = [δZlp1_δBl*δAlp1_δZlp1*δC_δAlp1] + nabla_B

            nabla_W = [np.dot(δAlp1_δZlp1*δC_δAlp1, δZlp1_δWl.transpose())] + nabla_W

            if self.verbose:
                print('nabla_B component: ', δZlp1_δBl*δAlp1_δZlp1*δC_δAlp1)
                print('nabla_W component: ', np.dot(δAlp1_δZlp1*δC_δAlp1, δZlp1_δWl.transpose()))
                print('self.W[l].shape:  ',self.W[l].shape)
                print('nabla_W[l].shape: ',nabla_W[0].shape)
                print()

            δC_δAlp1 = np.dot(self.W[l].transpose(), δAlp1_δZlp1*δC_δAlp1)

        
        self.feedbackCount = self.feedbackCount + 1

        for l in range(len(nabla_B)):
            self.nabla_W[l] = self.nabla_W[l] + nabla_W[l]
            self.nabla_B[l] = self.nabla_B[l] + nabla_B[l]

        # apply changes if batchsize has been reached
        if (self.feedbackCount >= self.batchSize) and applyChanges:
            
            for l in range(len(self.nabla_W)):
                self.W[l] = self.W[l] - self.nabla_W[l]*self.learningRate/self.batchSize
                self.B[l] = self.B[l] - self.nabla_B[l]*self.learningRate/self.batchSize

            self.nabla_W = [W*0 for W in self.W]
            self.nabla_B = [B*0 for B in self.B]
            self.feedbackCount = 0
        
        return δC_δAlp1

    def flush(self):
        self.nabla_W = [W*0 for W in self.W]
        self.nabla_B = [B*0 for B in self.B]
        self.A = [A*0 for A in self.A]

# %%
# (1.2) CNN base class

class CNN(MLModule):
    def zero(self, **kwargs):
        # print('kwargs: ', kwargs)

        crucialIncludes = ['filterSizes']
        for k in crucialIncludes:
            if k not in kwargs.keys():
                raise Exception(k + ' not included in keyed args. must include: ', crucialIncludes)
            else:
                exec('self.' + k + ' = '+str(kwargs[k]))


        # optional includes
        optionalIncludes = {'learningRate': 0.05, 'batchSize': 10, 'progressiveLearning': False , 'activation': 'linear', 'activation_prime': 'linear_prime'}
        for (k, v) in optionalIncludes.items():
            # print(k, v)
            if k not in kwargs.keys():
                # print('self.' + k + ' = '+str(v))
                exec('self.' + k + ' = '+str(v))
            else:
                exec('self.' + k + ' = '+str(kwargs[k]))
        

        self.feedbackCount = 0

        # A[l+1] = convolve(A[l], H[l])

        self.H = [np.random.randn(s, s) for s in self.filterSizes]
        # self.H = [normalise(np.random.randn(s, s)) for s in self.filterSizes]
        self.A = []
        self.Z = []

        self.nabla_H = [H*0 for H in self.H]
        self.learningFocus = 0

    def output(self):
        return self.A[-1]

    def feedForward(self):
        
        self.Z = []
        for l in range(len(self.H)):
            self.Z = self.Z + [conv(self.A[l], self.H[l])]
            self.A = self.A + [self.activation(self.Z[l])]

            if self.verbose:
                print('l: ', l)
                print('self.A[l].shape: ', self.A[l].shape)
                print('self.H[l].shape: ', self.H[l].shape)
                print('self.Z[l].shape: ', self.Z[l].shape)
            
            
            
        # self.A = self.A + [self.activation(self.Z[-1])]

        return self.output()
    
    def push(self, input):
        self.Z = []
        self.A = [input]
        return self.feedForward()

    def cost_derivative(self, Y_ideal):
        Y_ideal = np.reshape(Y_ideal,self.A[-1].shape)
                
        return (self.A[-1]-Y_ideal)

    def acceptFeedback(self, feedback, applyChanges = True):
        
        δC_δAlp1 = feedback - self.A[-1]
        return self.acceptNabla(δC_δAlp1, applyChanges = applyChanges)
    
    def acceptNabla(self, δC_δAlp1, applyChanges = True):
        # δC_δAlp1 = np.flip(δC_δAlp1)
        # if np.sum(abs(δC_δAlp1))>9999999999999999:
        #     return 0
        # else:
        #     print(np.sum(np.abs(δC_δAlp1)))
        #     print('np.sum(np.abs(δC_δAlp1))<0.001: ', np.sum(np.abs(δC_δAlp1))<0.001)

        nabla_H = []
        
        # backprop algorithm
        for l in list(range(len(self.H)))[::-1]:

            
            # ΔY = (Y - Y_ideal)
            # ΔH_ideal = (H - H_ideal)
            # ΔH = deconvolve(ΔY, X)
            if self.verbose:
                print('l: ', l)
                print('δC_δAlp1.shape:    ', δC_δAlp1.shape)
                print('self.A[l+1].shape: ', self.A[l+1].shape)
                print('self.Z[l].shape: ', self.Z[l].shape)
                print('self.A[l].shape:   ', self.A[l].shape)
                print('self.H[l].shape:   ', self.H[l].shape)

            ####        
            # ΔYc = (Y - Y_ideal)
            
            # ΔHc = deconvolve(ΔYc, X)
            ####
            ΔAΔZ = (δC_δAlp1*self.activation_prime(self.Z[l]))
            if self.verbose:
                print('******')
                print('ΔAΔZ.shape:        ', ΔAΔZ.shape)
                print('δC_δAlp1.shape:    ', δC_δAlp1.shape)
                print('self.A[l].shape:   ', self.A[l].shape)

            # nabla_H = [deconvolve(ΔAΔZ, self.A[l])] + nabla_H
            
            # δC_δAlp1 = deconvolve(ΔAΔZ, self.H[l])

            # TODO: for the love of christ I only need the following 2 lines of code to fix this
            #       I give up. for now...
            nabla_H = [iconv(ΔAΔZ, self.A[l])] + nabla_H
            δC_δAlp1 = iconv(ΔAΔZ, self.H[l])

            # nabla_H = [signal.convolve2d(ΔAΔZ, self.A[l],mode = 'valid')] + nabla_H
            # δC_δAlp1 = signal.convolve2d(ΔAΔZ, self.H[l],mode = 'full')

            if self.verbose:
                print('******')
                print('ΔAΔZ.shape:        ', ΔAΔZ.shape)
                print('δC_δAlp1.shape:    ', δC_δAlp1.shape)
                print('self.A[l].shape:   ', self.A[l].shape)
                print('******')
                print()

        self.feedbackCount = self.feedbackCount + 1

        for l in range(len(nabla_H)):
            # print('nabla_H[l].shape:', nabla_H[l].shape)
            # print('self.nabla_H[l].shape:', self.nabla_H[l].shape)
            self.nabla_H[l] = self.nabla_H[l] + nabla_H[l]

        # apply changes if batchsize has been reached
        if (self.feedbackCount >= self.batchSize) and applyChanges:
            
            for l in range(len(self.nabla_H)):
                if (self.learningFocus == l) or (not self.progressiveLearning):
                    self.H[l] = self.H[l] + self.nabla_H[l]*self.learningRate/self.feedbackCount
            self.learningFocus = (self.learningFocus+1)%len(self.nabla_H)
            self.nabla_H = [H*0 for H in self.H]
            self.feedbackCount = 0
        
        return δC_δAlp1

    def flush(self):
        self.nabla_H = [H*0 for H in self.H]
        self.A = []
        
    def plot(self):

        f, axarr = plt.subplots(len(self.A),4)
        for i in range(len(self.H)):


            axarr[i, 0].imshow(self.A[i], interpolation='none', cmap='Greys')
            axarr[i, 0].set_title('A['+str(i)+']')
            axarr[i, 1].imshow(self.Z[i], interpolation='none', cmap='Greys')
            axarr[i, 1].set_title('Z['+str(i)+']')
            axarr[i, 2].imshow(self.H[i], interpolation='none', cmap='Greys')
            axarr[i, 2].set_title('H['+str(i)+']')
            axarr[i, 3].imshow(self.nabla_H[i], interpolation='none', cmap='Greys')
            axarr[i, 3].set_title('ΔH['+str(i)+']')
            
        axarr[-1, 0].imshow(self.A[i+1], interpolation='none', cmap='Greys')
        axarr[-1, 0].set_title('A['+str(i+1)+']')
        
        plt.show()

# %%
# (1.3) ZNN base class

class ZNN(MLModule):
    def zero(self, **kwargs):
        # print('kwargs: ', kwargs)

        crucialIncludes = ['layerSizes']
        for k in crucialIncludes:
            if k not in kwargs.keys():
                raise Exception(k + ' not included in keyed args. must include: ', crucialIncludes)
            else:
                exec('self.' + k + ' = '+str(kwargs[k]))


        # optional includes
        optionalIncludes = {'learningRate': 0.05, 'batchSize': 10, 'activation': 'sigmoid', 'activation_prime': 'sigmoid_prime'}
        for (k, v) in optionalIncludes.items():
            # print(k, v)
            if k not in kwargs.keys():
                # print('self.' + k + ' = '+str(v))
                exec('self.' + k + ' = '+str(v))
            else:
                exec('self.' + k + ' = '+str(kwargs[k]))
        
        self.feedbackCount = 0

        # A[l+1] = Γ(W[l].A[l] + B[l])

        self.Z = [np.zeros((s, 1)) for s in self.layerSizes]
        
        self.B = [np.random.randn(s, 1) for s in self.layerSizes[1:]]
        self.W = [np.random.randn(self.layerSizes[i+1],self.layerSizes[i]) for i in range(len(self.layerSizes[:-1]))]    # TODO: Neurons should only connect to a small number of other neurons, not all. could fix w/ fimple filter func
                       
        self.nabla_W = [W*0 for W in self.W]
        self.nabla_B = [B*0 for B in self.B]

    def output(self):
        return self.Z[-1]

    def feedForward(self):
        for l in range(len(self.Z[:-1])):
            if self.verbose:
                print('l:', l)
                print('self.A[l+1].shape:', self.Z[l+1].shape)
                print('self.W[l].shape:', self.W[l].shape)
                print('self.A[l].shape:', self.Z[l].shape)
                print('self.B[l].shape:', self.B[l].shape)
            A = self.activation(self.Z[l])
            self.Z[l+1] = np.dot(self.W[l], A)+self.B[l]

        return self.output()
    
    def push(self, input):
        self.Z[0] = np.reshape(input,self.Z[0].shape)
        return self.feedForward()

    def cost_derivative(self, Y_ideal):
        Y_ideal = np.reshape(Y_ideal,self.Z[-1].shape)
                
        return (self.Z[-1]-Y_ideal)

    def acceptFeedback(self, feedback, applyChanges = True):
        
        δC_δZlp1 = self.cost_derivative(feedback)
        return self.acceptNabla(δC_δZlp1, applyChanges = applyChanges)
    
    def acceptNabla(self, δC_δZlp1, applyChanges = True):

        nabla_B = []
        nabla_W = []

        # backprop algorithm
        for l in list(range(len(self.B)))[::-1]:
                
            
            δZlp1_δWl = self.activation(self.Z[l])
            δZlp1_δBl = 1



            if self.verbose:
                print('l: ', l)
                print('δC_δAlp1.shape: ',δC_δAlp1.shape)
                print('δAlp1_δZlp1.shape: ',δAlp1_δZlp1.shape)
                print('δZlp1_δWl.shape: ',δZlp1_δWl.shape)
                print('δC_δAlp1:    ',δC_δAlp1)
                print('δAlp1_δZlp1: ',δAlp1_δZlp1)
                print('δZlp1_δWl:   ',δZlp1_δWl)
                # print('δZlp1_δBl.shape: ',δZl_δBl.shape)
                print()
            
            nabla_B = [δZlp1_δBl*δC_δZlp1] + nabla_B

            nabla_W = [np.dot(δC_δZlp1, δZlp1_δWl.transpose())] + nabla_W

            if self.verbose:
                print('nabla_B component: ', δZlp1_δBl*δAlp1_δZlp1*δC_δAlp1)
                print('nabla_W component: ', np.dot(δAlp1_δZlp1*δC_δAlp1, δZlp1_δWl.transpose()))
                print('self.W[l].shape:  ',self.W[l].shape)
                print('nabla_W[l].shape: ',nabla_W[0].shape)
                print()

            # for next iteration: 
            

            δC_δAlp1 = np.dot(self.W[l].transpose(), δC_δZlp1)
            #Zlp1 = np.dot(self.W[l-1], self.A[l-1]) + self.B[l-1]
            δAlp1_δZlp1 = self.activation_prime(self.Z[l])
            δC_δZlp1 = δAlp1_δZlp1*δC_δAlp1

        
        self.feedbackCount = self.feedbackCount + 1

        for l in range(len(nabla_B)):
            self.nabla_W[l] = self.nabla_W[l] + nabla_W[l]
            self.nabla_B[l] = self.nabla_B[l] + nabla_B[l]

        # apply changes if batchsize has been reached
        if (self.feedbackCount >= self.batchSize) and applyChanges:
            
            for l in range(len(self.nabla_W)):
                self.W[l] = self.W[l] - self.nabla_W[l]*self.learningRate/self.batchSize
                self.B[l] = self.B[l] - self.nabla_B[l]*self.learningRate/self.batchSize

            self.nabla_W = [W*0 for W in self.W]
            self.nabla_B = [B*0 for B in self.B]
            self.feedbackCount = 0
        
        return δC_δZlp1

    def flush(self):
        self.nabla_W = [W*0 for W in self.W]
        self.nabla_B = [B*0 for B in self.B]
        self.Z = [Z*0 for Z in self.Z]

# %%
# (1.4) GNN base class

class GNN(MLModule):
    def zero(self, **kwargs):
        # print('kwargs: ', kwargs)

        crucialIncludes = ['inputSize', 'outputSize', 'hiddenSize']
        for k in crucialIncludes:
            if k not in kwargs.keys():
                raise Exception(k + ' not included in keyed args. must include: ', crucialIncludes)


        # optional includes
        
        for (k, v) in {'learningRate': 0.05, 'numTrainingIterations': 1, 'batchSize': 10, 'activation': 'sigmoid', 'activation_prime': 'sigmoid_prime'}.items():
            # print(k, v)
            if k not in kwargs.keys():
                print('self.' + k + ' = '+str(v))
                exec('self.' + k + ' = '+str(v))
            else:
                exec('self.' + k + ' = '+str(kwargs[k]))
               
        totalSize = kwargs['inputSize'] + kwargs['outputSize'] + kwargs['hiddenSize'] + 1
        
        # self.A = [np.zeros((totalSize, 1))+0.5 for i in range(self.numTrainingIterations)]
        self.A = [np.zeros((totalSize, 1))for i in range(self.numTrainingIterations)]
        
        
        # self.B = np.random.randn(totalSize, 1)*0.01       # TODO: include as constant weights and activation
        self.W = (np.random.randn(totalSize,totalSize)/2)    # TODO: Neurons should only connect to a small number of other neurons, not all. could fix w/ fimple filter func
        self.W = (np.ones_like(self.W) - np.eye(totalSize)*0.99 - np.triu(np.ones_like(self.W), 1)*0.99)*self.W # include this to reduce the initial size of the retention and feedback values

        self.inputSlice = slice(0,kwargs['inputSize'])
        self.outputSlice = slice(totalSize - kwargs['outputSize'] - 1, totalSize-1)
        self.biasSlice = slice(totalSize-1, totalSize)

        # biases
        self.W[:, self.biasSlice] = (np.random.randn(totalSize,1)-0.5)*0.01
        
        # inputs, biases should have unidirectional connections
        self.W[:kwargs['inputSize']] = self.W[:kwargs['inputSize']]*0
        self.W[self.biasSlice, :] = 0
        self.A[-1][self.biasSlice]=1
        self.W[self.biasSlice, self.biasSlice] = 1
        # self.B[:kwargs['inputSize']] = self.B[:kwargs['inputSize']]*0

        # outputs should be independent from one another
        self.W[self.outputSlice, self.outputSlice] = 0
        self.nabla_W = []
        # self.nabla_B = []
        return self
    
    def push(self, input):
        input = np.reshape(input,self.A[-1][self.inputSlice].shape)
        self.A[-1][self.inputSlice]=input
        return self.tick()

    # perform one feed forward iteration
    def tick(self):
        
        self.A[-1][self.biasSlice] = 1

        Z = np.dot(self.W, self.A[-1])
        
        self.A = self.A[1:] + [self.activation(Z)]

        return self.output()
    
    def output(self):
        return self.A[-1][self.outputSlice]

    def cost_derivative(self, Y_ideal):
        Y_ideal = np.reshape(Y_ideal,self.A[-1][self.outputSlice].shape)
        
        nabla_A = np.zeros_like(self.A[-1])
        nabla_A[self.outputSlice] = Y_ideal
        
        nabla_A[self.outputSlice] = (self.A[-1]-nabla_A)[self.outputSlice]
        
        return nabla_A

    def acceptFeedback(self, Y_ideal):
        
        
        if len(self.A) >= self.numTrainingIterations:
            
            δC_δAl = self.cost_derivative(Y_ideal)
            # print('δC_δAl: ',δC_δAl)
            
            for i in range(len(self.A)-self.numTrainingIterations, len(self.A))[::-1]:
                
                Z = np.dot(self.W, self.A[i-1])
                δAl_δZl = self.activation_prime(Z)
                δZl_δW = self.A[i-1]
                δZl_δB = 1



                #self.nabla_B = self.nabla_B + [δZl_δB*δAl_δZl*δC_δAl*(self.B!=0)]

                self.nabla_W = self.nabla_W + [np.dot(δAl_δZl*δC_δAl, δZl_δW.transpose())*(self.W!=0)]

                δC_δAl = np.dot(self.W.transpose(), δAl_δZl*δC_δAl)
                
                # print('self.A[i]: ', self.A[i])
                # print('self.A[i-1]: ', self.A[i-1])
                # print('Z: ', Z)
                # print('δAl_δZl: ', δAl_δZl)
                # print('δZl_δW: ', δZl_δW)
                # print('δZl_δB: ', δZl_δB)
                # print('nabla_B: ', nabla_B )
                #print('self.nabla_W[-1]: ', self.nabla_W[-1])
                #print('self.W: ', self.W)
                # print('δC_δAl: ', δC_δAl)
                # print()
                # print('Z: ', Z)
                # print('sp: ', sp)
                # print('nabla_A: ', nabla_A)
                # print('self.A: ', self.A[i].reshape((1,len(self.A[i]))))
                # print('nabla_B: ', nabla_B)
                # print('self.W: ', self.W)
                # print('nabla_W: ', nabla_W)

                # Z = np.dot(self.W, self.A[i-1]) + self.B
                # nabla_B = self.activation_prime(Z)*nabla_A[-i-1]
                # nabla_W = np.dot(self.activation_prime(Z)*nabla_A[-i-1], self.A[-i-1].transpose())


        if len(self.nabla_W) >=self.batchSize:
            # print('*********************************')
            # print('self.W: ', self.W)
            # print('self.B: ', self.B)
            for i, nb in enumerate(self.nabla_W):
                self.W = self.W - self.nabla_W[i]*self.learningRate
                #self.B = self.B - self.nabla_B[i]*self.learningRate

            #self.nabla_B = []
            self.nabla_W = []
            # print('self.W: ', self.W)
        
        return self