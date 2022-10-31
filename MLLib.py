# %%
# (0.0) Imports, inits and TODOs
# TODO: next major version will have:
        # - python logging instead of printing/verbosity
        # - __init__ funcs which use the super.__init__() function so args/optional args are explicit
        # - unit testing as it's being built
# TODO: calculate variance in Δparam in each batch. split largest variance node into 2 nodes, one with the positive applied, one with negative
# TODO: wrapper module
# TODO: comment everything
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
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

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

def difference(Y, Y_ideal):
    Y_ideal = np.reshape(Y_ideal,Y.shape)
    return Y - Y_ideal

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
    def __init__(self, fromStr = '', comment = '', logLevel = logging.INFO):
        
        if fromStr != '':
            self.fromStr(fromStr)
            
        else:
            self.comment = comment
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logLevel)

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
            # case "<class 'RootLogger'>":
            #     retval = 'logger.getLogger('+__name__+')'
            case "<class 'logging.Logger'>":
                #print(str('serialising logger:',v))
                retval = 'logging.getLogger('+__name__+')'

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

    # to be overridden: takes ideal output (feedback) for current state. uses it to learn
    def acceptFeedback(self, feedback):
        return feedback

    # to be overridden: takes ideal output (feedback) for current state. uses it to learn
    def acceptNabla(self, nabla):
        return nabla

    # to be overridden: module may need to meet initialiation criteria after flush
    def isInitialised(self):
        return True

    # so obj can be used as dict key
    def hash(self):
        return(hash(str(self)))

# %%
# (1.1) NN base class

class NN(MLModule):

    # initialise class
    def __init__(self, layerSizes, fromStr = '', comment = '', logLevel = logging.INFO, learningRate = 0.05, batchSize = 10, activation = sigmoid, activation_prime = sigmoid_prime, cost_derivative = difference):
        
        # initialise all instance specific params

        self.layerSizes = layerSizes

        self.learningRate = learningRate
        self.batchSize = batchSize
        self.activation = activation
        self.activation_prime = activation_prime
        self.cost_derivative = cost_derivative

        # every batchSize feedbacks, apply changes
        self.feedbackCount = 0

        # A[l+1] = Γ(W[l].A[l] + B[l])
        self.A = [np.zeros((s, 1)) for s in self.layerSizes]
        self.B = [np.random.randn(s, 1) for s in self.layerSizes[1:]]
        self.W = [np.random.randn(self.layerSizes[i+1],self.layerSizes[i]) for i in range(len(self.layerSizes[:-1]))]

        # store changes
        self.nabla_W = [W*0 for W in self.W]
        self.nabla_B = [B*0 for B in self.B]

        # initialise universal params. will overwrite from string here if applicable
        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)

    # return output of last feedForward
    def output(self):
        return self.A[-1]

    # one pass of feedForward algorithm
    def feedForward(self):
        for l in range(len(self.A[:-1])):

            self.logger.debug('l:', l)
            self.logger.debug('self.A[l+1].shape:', self.A[l+1].shape)
            self.logger.debug('self.W[l].shape:', self.W[l].shape)
            self.logger.debug('self.A[l].shape:', self.A[l].shape)
            self.logger.debug('self.B[l].shape:', self.B[l].shape)
            
            self.A[l+1] = self.activation(np.dot(self.W[l], self.A[l])+self.B[l])

        return self.output()

    # take input and feedForward
    def push(self, input):
        self.A[0] = np.reshape(input,self.A[0].shape)
        return self.feedForward()

    # accept Y_ideal, generate changes
    def acceptFeedback(self, feedback, applyChanges = True):
        
        δC_δAlp1 = self.cost_derivative(self.A[-1], feedback)
        return self.acceptNabla(δC_δAlp1, applyChanges = applyChanges)
    
    # accept ΔY, generate changes
    def acceptNabla(self, δC_δAlp1, applyChanges = True):

        nabla_B = []
        nabla_W = []

        # backprop algorithm
        for l in list(range(len(self.B)))[::-1]:
                
            Zlp1 = np.dot(self.W[l], self.A[l]) + self.B[l]
            δAlp1_δZlp1 = self.activation_prime(Zlp1)
            δZlp1_δWl = self.A[l]
            δZlp1_δBl = 1


            self.logger.debug('l: ', l)
            self.logger.debug('δC_δAlp1.shape: ',δC_δAlp1.shape)
            self.logger.debug('δAlp1_δZlp1.shape: ',δAlp1_δZlp1.shape)
            self.logger.debug('δZlp1_δWl.shape: ',δZlp1_δWl.shape)
            self.logger.debug('δC_δAlp1:    ',δC_δAlp1)
            self.logger.debug('δAlp1_δZlp1: ',δAlp1_δZlp1)
            self.logger.debug('δZlp1_δWl:   ',δZlp1_δWl)
            
            
            nabla_B = [δZlp1_δBl*δAlp1_δZlp1*δC_δAlp1] + nabla_B
            nabla_W = [np.dot(δAlp1_δZlp1*δC_δAlp1, δZlp1_δWl.transpose())] + nabla_W


            self.logger.debug('nabla_B component: ', δZlp1_δBl*δAlp1_δZlp1*δC_δAlp1)
            self.logger.debug('nabla_W component: ', np.dot(δAlp1_δZlp1*δC_δAlp1, δZlp1_δWl.transpose()))
            self.logger.debug('self.W[l].shape:  ',self.W[l].shape)
            self.logger.debug('nabla_W[l].shape: ',nabla_W[0].shape)


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

    # reset all nablas, activations to 0
    def flush(self):
        self.nabla_W = [W*0 for W in self.W]
        self.nabla_B = [B*0 for B in self.B]
        self.A = [A*0 for A in self.A]

    
