# %%
# (0.0) TODOs
# THIS VERSION:
# TODO: comment everything
# TODO: test everything
# TODO: input function on everything
# TODO: downsample/pooling module
# TODO: modernise and fix CNN
# TODO: change all prints to logs
# TODO: implement obj.setLogLevel(l), obj.info(msg)/obj.debug(msg)/... 
    # - setting log level will set levels below l to a pass statement equivalent, cutting down on compute time rather than getting logger to try/decide not to log everything

# SUBSEQUENT VERSIONS:
# TODO: module series: get output layer at end of some layer l
# TODO: function layer performing a single function on all inputs
# TODO: merge empirical model and function into a base class w/ 2 interfaces: __call__(), __getitem__()
# TODO: funnel module to push output to multiple different places
# TODO: change cost derivatives to variable functions. define at init
# TODO: better learning than backprop
    # - if gradient batch diverges (has two humps), split neuron in 2 and apply gradient of each hump so that the neuron does both things backprop is asking
        # - calculate variance in Δparam in each batch. split largest variance node into 2 nodes, one with the positive applied, one with negative
    # - component-wise learning
        # - output vector is made of components f'ed together
        # - identify components
        # - use grad to identify most faulty component
        # - improve that component only
        # - see Z as a stack of componenets rather than 1 value
    # - learning rate proportional to variance
        # - if all Δ values in batch are the same, LR should be 1. reduces from there
# TODO: stride in convolution funcs
# TODO: GAN flush function should run it with 0 input until it converges-if it converges
# TODO: add option in backprop to randomise lowest contributing/wrongly contributing nodes
# TODO: Tokeniser/encoder/transformer classes
# TODO: stable diffusion network
# TODO: training scheme definitions
# TODO: reinforcement learning infrastructure
# TODO: vector word embeddings
    # - scrape wikipedia intros for sentences
# TODO: port checkers game for reinforcement learning
# TODO: make snake game for reinforcement learning

# %%
# (0.1) imports/initialisations

import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import datetime

# warnings.simplefilter('error')
np.set_printoptions(linewidth = 1000, suppress=True)

# %%
# (0.2) basic function defs

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

def zipFuncs(funcs):
    def retFunc(z):
        y = z*0
        for i, f in enumerate(funcs):
            y[i::len(funcs)] = f(z[i::len(funcs)])
        return y
    return retFunc

def difference(Y, Y_ideal):
    Y_ideal = np.reshape(Y_ideal,Y.shape)
    return Y - Y_ideal

def argmax_prime(Y, Y_ideal):
    digit = np.argmax(Y_ideal)
    return (Y>Y[digit])*(Y[digit]-Y) + (Y.max()-Y[digit])*(Y_ideal==Y_ideal.max())

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
# (0.3) basic obj that can be easily printed/saved/loaded 

class basicObj():
    # init function for every MLModule child
    def __init__(self, fromStr = '', comment = '', logLevel = logging.WARNING):
        
        if fromStr != '':
            self.fromStr(fromStr)
            
        else:
            self.comment = comment
            self.id = hash(str(datetime.datetime.now())+str(self.__class__.__name__))
            self.logger = logging.getLogger(str(self.__class__.__name__)+'_'+str(self.id))
            self.logger.setLevel(logLevel)
            
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # # create file handler which logs even debug messages
            # fh = logging.FileHandler('spam.log')
            # fh.setLevel(logging.DEBUG)
            # fh.setFormatter(formatter)
            # self.logger.addHandler(fh)
            pass

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
     
    # so obj can be used as dict key
    def hash(self):
        return(hash(str(self)))


# %%
# (1.0) MLModule Base Class

class MLModule(basicObj):

    # to be overridden: push input, return input
    def input(self, input):
        return input

    # to be overridden: push input, pop output
    def push(self, input):
        return self.feedforward()

    # to be overridden: feed forward from stored input, return output
    def feedforward(self):
        return self.output()

    # to be overridden: maintain structure, flush IO dependent vars + nablas with 0s
    def flush(self):
        return self

    # to be overridden: return output
    def output(self):
        return np.array([])

    # to be overridden: takes ideal output (feedback) for current state. uses it to learn
    def acceptFeedback(self, feedback):
        return feedback

    # to be overridden: takes ideal output (feedback) for current state. uses it to learn
    def acceptNabla(self, nabla):
        return nabla

    # to be overridden: module may need to meet initialiation criteria after flush
    def isInitialised(self):
        return True


# %%
# (1.1) Parallel Module Base Class

class ModuleParallel(MLModule):

    # initialise class
    def __init__(self, modules, fromStr = '', comment = '', logLevel = logging.WARNING):

        self.logger = None

        # modules follow format [a,[b,c],d] where input X is fed into a, module series[b,c], d which each create component of output Y
        self.modules = []

        # get output shape and slices associated with each module
        self.outputSlices = []
        self.inputSlices = []
        xTally = 0
        yTally = 0
        xMax = 0
        yMax = 0
        for i, m in enumerate(modules):
            
            if str(type(m)) == "<class 'list'>":
                m = ModuleSeries(m, fromStr = fromStr, comment = comment, logLevel = logLevel)

            outShape = m.output().shape
            self.outputSlices = self.outputSlices + [(slice(yTally,yTally + outShape[0]),slice(0,outShape[1]))]
            yTally = yTally + outShape[0]
            yMax = max(yMax,outShape[1])

            inShape = m.input(None).shape
            self.inputSlices = self.inputSlices + [(slice(xTally,xTally + inShape[0]),slice(0,inShape[1]))]
            xTally = xTally + inShape[0]
            xMax = max(yMax,outShape[1])

            self.modules = self.modules + [m]

        self.outputShape = (yTally,yMax)
        self.inputShape  = (xTally,xMax)

        # initialise the default variables
        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)
        for m in self.modules:
            m.logger = self.logger

    # return output of last feedForward
    def output(self):
        output = np.zeros(self.outputShape)
        for m, s in zip(self.modules,self.outputSlices):
            output[s] = m.output()
        return output

    # one pass of feedForward algorithm
    def feedForward(self):
        for m in self.modules:
            m.feedForward()
        return self.output()

    # take and return an input
    def input(self, input):
        if type(input) != type(None):
            self.A[0] = np.reshape(input,self.A[0].shape)
            retInput = np.zeros(self.inputShape)
            for m, s in zip(self.modules,self.inputSlices):
                retInput[s] = m.input(input)

            return retInput
        
        retInput = np.zeros(self.inputShape)
        for m, s in zip(self.modules,self.inputSlices):
            retInput[s] = m.input(input[s])

        return retInput

    # take input and feedForward
    def push(self, input):
        for m, s in zip(self.modules,self.inputSlices):
            m.push(input[s])
        return self.output()

    # accept Y_ideal, generate changes
    def acceptFeedback(self, feedback, applyChanges = True):
        ΔX = np.zeros(self.inputShape)
        for m, si, so in zip(self.modules,self.inputSlices, self.outputSlices):
            ΔX[si] = ΔX[si] + m.acceptFeedback(feedback[so])
        return ΔX

    # accept ΔY, generate changes
    def acceptNabla(self, ΔY, applyChanges = True):
        
        ΔX = np.zeros(self.inputShape)
        for m, si, so in zip(self.modules,self.inputSlices, self.outputSlices):
            ΔX[si] = ΔX[si] + m.acceptNabla(ΔY[so])
        return ΔX

    # reset all nablas, activations to 0
    def flush(self):
        for m in self.modules:
            m.flush()
        return self


# %%
# (1.2) MLWrapper Base Class

class ModuleSeries(MLModule):

    # initialise class
    def __init__(self, modules, fromStr = '', comment = '', logLevel = logging.WARNING):

        # modules follow format [a,[b,c],d] where output of a feeds into b,c in parallel, the outputs of which feed into d
        self.modules = []


        for i, m in enumerate(modules):
            
            if str(type(m)) == "<class 'list'>":
                m = ModuleParallel(m, fromStr = fromStr, comment = comment, logLevel = logLevel)
            
            self.modules = self.modules + [m]

        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)
        for m in self.modules:
            m.logger = self.logger

    # return output of specified feedForward
    def output(self, layer = -1):
        return self.modules[layer].output()

    # one pass of feedForward algorithm
    def feedForward(self):
        X = self.modules[0].feedForward()
        for m in self.modules[1:]:
            X = m.push(X)
        return X

    # take and return input
    def input(self, input):
        return self.modules[0].input(input)

    # take input and feedForward
    def push(self, input):
        self.input(input)
        return self.feedForward()

    # accept Y_ideal, generate changes
    def acceptFeedback(self, feedback, applyChanges = True):

        ΔY = self.modules[-1].acceptFeedback(feedback, applyChanges = True)

        for m in self.modules[::-1][1:]:
            ΔY = m.acceptNabla(ΔY)

        return ΔY
    
    # accept ΔY, generate changes
    def acceptNabla(self, ΔY, applyChanges = True):

        for m in self.modules[::-1]:
            ΔY = m.acceptNabla(ΔY)

        return ΔY

    # reset all nablas, activations to 0
    def flush(self):
        for m in self.modules:
            m.flush()
        return self


# %%
# (1.3) NN base class

class NN(MLModule):

    # initialise class
    def __init__(self, layerSizes, fromStr = '', comment = '', logLevel = logging.WARNING, learningRate = 0.05, batchSize = 10, activation = sigmoid, activation_prime = sigmoid_prime, cost_derivative = difference):
        
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
        self.Z = [np.zeros((s, 1)) for s in self.layerSizes]
        self.B = [np.random.randn(s, 1) for s in self.layerSizes[1:]]
        self.W = [np.random.randn(self.layerSizes[i+1],self.layerSizes[i]) for i in range(len(self.layerSizes[:-1]))]

        # store changes

        self.nabla_B = [B*0 for B in self.B]
        self.nabla_W = [W*0 for W in self.W]
        
        # initialise universal params. will overwrite from string here if applicable
        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)

    # return output of last feedForward
    def output(self):
        return self.A[-1]

    # one pass of feedForward algorithm
    def feedForward(self):
        for l in range(len(self.A[:-1])):

            # self.logger.debug('l:'+ str(l))
            # self.logger.debug('self.A[l+1].shape:'+ str(self.A[l+1].shape))
            # self.logger.debug('self.W[l].shape:'+ str(self.W[l].shape))
            # self.logger.debug('self.A[l].shape:'+ str(self.A[l].shape))
            # self.logger.debug('self.B[l].shape:'+ str(self.B[l].shape))
            self.Z[l+1] = np.dot(self.W[l], self.A[l])+self.B[l]
            self.A[l+1] = self.activation(self.Z[l+1])

        return self.output()

    # take and return an input
    def input(self, input):
        if type(input) != type(None):
            self.A[0] = np.reshape(input,self.A[0].shape)
        return self.A[0]

    # take input and feedForward
    def push(self, input):
        self.input(input)
        return self.feedForward()

    # accept Y_ideal, generate changes
    def acceptFeedback(self, feedback, applyChanges = True):
        
        ΔA = self.cost_derivative(self.A[-1], feedback)
        return self.acceptNabla(ΔA, applyChanges = applyChanges)
    
    # accept ΔY, generate changes
    def acceptNabla(self, ΔA, applyChanges = True):

        
        nabla_B = []
        nabla_W = []

        # backprop algorithm
        for l in list(range(len(self.A)))[::-1][:-1]:
                
            
            ΔA_ΔZ = self.activation_prime(self.Z[l])
            ΔZ = ΔA*ΔA_ΔZ
            ΔZ_ΔWm1 = self.A[l-1]
            ΔZ_δBm1 = 1


            # self.logger.debug('l: '+ str(l))
            # self.logger.debug('ΔA.shape: '+str(ΔA.shape))
            # self.logger.debug('ΔA_ΔZ.shape: '+str(ΔA_ΔZ.shape))
            # self.logger.debug('ΔZ_ΔWm1.shape: '+str(ΔZ_ΔWm1.shape))
            # self.logger.debug('ΔA:    '+str(ΔA))
            # self.logger.debug('ΔA_ΔZ: '+str(ΔA_ΔZ))
            # self.logger.debug('ΔZ_ΔWm1:   '+str(ΔZ_ΔWm1))
            
            
            nabla_B = [ΔZ] + nabla_B
            nabla_W = [np.dot(ΔZ, ΔZ_ΔWm1.transpose())] + nabla_W


            # self.logger.debug('nabla_B component: '+ str(ΔZ_δBm1*ΔZ))
            # self.logger.debug('nabla_W component: '+ str(np.dot(ΔZ, ΔZ_ΔWm1.transpose())))
            # self.logger.debug('self.W[l-1].shape:  '+str(self.W[l-1].shape))
            #self.logger.debug('nabla_W[l].shape: ',nabla_W[l].shape)


            # for next iteration
            ΔA = np.dot(self.W[l-1].transpose(), ΔZ)

        
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
        
        return ΔA

    # reset all nablas, activations to 0
    def flush(self):
        self.nabla_W = [W*0 for W in self.W]
        self.nabla_B = [B*0 for B in self.B]
        self.A = [A*0 for A in self.A]
        self.Z = [Z*0 for Z in self.Z]


# %%
# (1.4) ZNN base class

class ZNN(MLModule):

    # initialise class
    def __init__(self, layerSizes, fromStr = '', comment = '', logLevel = logging.WARNING, learningRate = 0.05, batchSize = 10, activation = sigmoid, activation_prime = sigmoid_prime, cost_derivative = difference):
        
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
        self.Z = [np.zeros((s, 1)) for s in self.layerSizes]
        self.B = [np.random.randn(s, 1) for s in self.layerSizes[1:]]
        self.W = [np.random.randn(self.layerSizes[i+1],self.layerSizes[i]) for i in range(len(self.layerSizes[:-1]))]

        # store changes

        self.nabla_B = [B*0 for B in self.B]
        self.nabla_W = [W*0 for W in self.W]
        
        # initialise universal params. will overwrite from string here if applicable
        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)

    # return output of last feedForward
    def output(self):
        return self.A[-1]

    # one pass of feedForward algorithm
    def feedForward(self):
        for l in range(len(self.A[:-1])):

            self.logger.debug('l:', str(l))
            self.logger.debug('self.A[l+1].shape:'+ str(self.A[l+1].shape))
            self.logger.debug('self.W[l].shape:'+ str(self.W[l].shape))
            self.logger.debug('self.A[l].shape:'+ str(self.A[l].shape))
            self.logger.debug('self.B[l].shape:'+ str(self.B[l].shape))
            self.Z[l+1] = np.dot(self.W[l], self.A[l])+self.B[l]
            self.A[l+1] = self.activation(self.Z[l+1])

        return self.output()

    # take and return an input
    def input(self, input):
        if type(input) != type(None):
            self.A[0] = np.reshape(input,self.A[0].shape)
        return self.A[0]

    # take input and feedForward
    def push(self, input):
        self.input(input)
        return self.feedForward()

    # accept Y_ideal, generate changes
    def acceptFeedback(self, feedback, applyChanges = True):
        
        ΔZ = self.cost_derivative(self.Z[-1], feedback)
        return self.acceptNabla(ΔZ, applyChanges = applyChanges)
    
    # accept ΔY, generate changes
    def acceptNabla(self, ΔZ, applyChanges = True):

        
        nabla_B = []
        nabla_W = []

        # backprop algorithm
        for l in list(range(len(self.A)))[::-1][:-1]:
                
            
            ΔZ_ΔWm1 = self.A[l-1]
            ΔZ_δBm1 = 1


            self.logger.debug('l: ', str(l))
            
            self.logger.debug('ΔZ_ΔWm1.shape: '+str(ΔZ_ΔWm1.shape))
            self.logger.debug('ΔZ_ΔWm1:   '+str(ΔZ_ΔWm1))
            
            
            nabla_B = [ΔZ] + nabla_B
            nabla_W = [np.dot(ΔZ, ΔZ_ΔWm1.transpose())] + nabla_W


            self.logger.debug('nabla_B component: '+ str(ΔZ_δBm1*ΔZ))
            self.logger.debug('nabla_W component: '+ str(np.dot(ΔZ, ΔZ_ΔWm1.transpose())))
            self.logger.debug('self.W[l-1].shape:  '+str(self.W[l-1].shape))
            #self.logger.debug('nabla_W[l].shape: ',nabla_W[l].shape)

            # for next iteration
            ΔA = np.dot(self.W[l-1].transpose(), ΔZ)
            ΔA_ΔZ = self.activation_prime(self.Z[l-1])
            ΔZ = ΔA*ΔA_ΔZ

            self.logger.debug('ΔA.shape: '+str(ΔA.shape))
            self.logger.debug('ΔA_ΔZ.shape: '+str(ΔA_ΔZ.shape))
            self.logger.debug('ΔA:    '+str(ΔA))
            self.logger.debug('ΔA_ΔZ: '+str(ΔA_ΔZ))
        
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
        
        return ΔA

    # reset all nablas, activations to 0
    def flush(self):
        self.nabla_W = [W*0 for W in self.W]
        self.nabla_B = [B*0 for B in self.B]
        self.A = [A*0 for A in self.A]
        self.Z = [Z*0 for Z in self.Z]

# %%
# (1.5) Function layer base class

class FL(MLModule):
        # initialise class
    def __init__(self, f, f_prime, fromStr = '', comment = '', logLevel = logging.WARNING, cost_derivative = difference):

        self.X = np.array([])
        self.ΔX = np.array([])
        self.f = f
        self.f_prime = f_prime
        self.cost_derivative = cost_derivative

        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)

    # push input, return input
    def input(self, input):
        if type(input) != type(None):
            self.X = input
        return self.X

    # feed forward from stored input, return output
    def feedForward(self):
        self.Y = self.f(self.X)
        return self.output()

    # push input, pop output
    def push(self, input):
        self.input(input)
        return self.feedForward()

    # maintain structure, flush IO dependent vars + nablas with 0s
    def flush(self):
        self.X = np.array([])
        return self

    # to be overridden: return output
    def output(self):
        return self.Y

    # to be overridden: takes ideal output (feedback) for current state. uses it to learn
    def acceptFeedback(self, feedback):
        return self.acceptNabla(self.cost_derivative(self.Y, feedback))

    # takes ideal output (feedback) for current state. uses it to learn
    def acceptNabla(self, nabla):
        return self.f_prime(nabla)



# %%
# (1.6) CNN base class

# class CNN(MLModule):
#     # TODO: test, modernise, rename to just reflect it only doing convolutions not pooling

#     def zero(self, **kwargs):
#         # print('kwargs: ', kwargs)

#         crucialIncludes = ['filterSizes']
#         for k in crucialIncludes:
#             if k not in kwargs.keys():
#                 raise Exception(k + ' not included in keyed args. must include: ', crucialIncludes)
#             else:
#                 exec('self.' + k + ' = '+str(kwargs[k]))


#         # optional includes
#         optionalIncludes = {'learningRate': 0.05, 'batchSize': 10, 'progressiveLearning': False , 'activation': 'linear', 'activation_prime': 'linear_prime'}
#         for (k, v) in optionalIncludes.items():
#             # print(k, v)
#             if k not in kwargs.keys():
#                 # print('self.' + k + ' = '+str(v))
#                 exec('self.' + k + ' = '+str(v))
#             else:
#                 exec('self.' + k + ' = '+str(kwargs[k]))
        

#         self.feedbackCount = 0

#         # A[l+1] = convolve(A[l], H[l])

#         self.H = [np.random.randn(s, s) for s in self.filterSizes]
#         # self.H = [normalise(np.random.randn(s, s)) for s in self.filterSizes]
#         self.A = []
#         self.Z = []

#         self.nabla_H = [H*0 for H in self.H]
#         self.learningFocus = 0

#     def output(self):
#         return self.A[-1]

#     def feedForward(self):
        
#         self.Z = []
#         for l in range(len(self.H)):
#             self.Z = self.Z + [conv(self.A[l], self.H[l])]
#             self.A = self.A + [self.activation(self.Z[l])]

#             if self.verbose:
#                 print('l: ', l)
#                 print('self.A[l].shape: ', self.A[l].shape)
#                 print('self.H[l].shape: ', self.H[l].shape)
#                 print('self.Z[l].shape: ', self.Z[l].shape)
            
            
            
#         # self.A = self.A + [self.activation(self.Z[-1])]

#         return self.output()
    
#     def push(self, input):
#         self.Z = []
#         self.A = [input]
#         return self.feedForward()

#     def cost_derivative(self, Y_ideal):
#         Y_ideal = np.reshape(Y_ideal,self.A[-1].shape)
                
#         return (self.A[-1]-Y_ideal)

#     def acceptFeedback(self, feedback, applyChanges = True):
        
#         δC_δAlp1 = feedback - self.A[-1]
#         return self.acceptNabla(δC_δAlp1, applyChanges = applyChanges)
    
#     def acceptNabla(self, δC_δAlp1, applyChanges = True):
#         # δC_δAlp1 = np.flip(δC_δAlp1)
#         # if np.sum(abs(δC_δAlp1))>9999999999999999:
#         #     return 0
#         # else:
#         #     print(np.sum(np.abs(δC_δAlp1)))
#         #     print('np.sum(np.abs(δC_δAlp1))<0.001: ', np.sum(np.abs(δC_δAlp1))<0.001)

#         nabla_H = []
        
#         # backprop algorithm
#         for l in list(range(len(self.H)))[::-1]:

            
#             # ΔY = (Y - Y_ideal)
#             # ΔH_ideal = (H - H_ideal)
#             # ΔH = deconvolve(ΔY, X)
#             if self.verbose:
#                 print('l: ', l)
#                 print('δC_δAlp1.shape:    ', δC_δAlp1.shape)
#                 print('self.A[l+1].shape: ', self.A[l+1].shape)
#                 print('self.Z[l].shape: ', self.Z[l].shape)
#                 print('self.A[l].shape:   ', self.A[l].shape)
#                 print('self.H[l].shape:   ', self.H[l].shape)

#             ####        
#             # ΔYc = (Y - Y_ideal)
            
#             # ΔHc = deconvolve(ΔYc, X)
#             ####
#             ΔAΔZ = (δC_δAlp1*self.activation_prime(self.Z[l]))
#             if self.verbose:
#                 print('******')
#                 print('ΔAΔZ.shape:        ', ΔAΔZ.shape)
#                 print('δC_δAlp1.shape:    ', δC_δAlp1.shape)
#                 print('self.A[l].shape:   ', self.A[l].shape)

#             # nabla_H = [deconvolve(ΔAΔZ, self.A[l])] + nabla_H
            
#             # δC_δAlp1 = deconvolve(ΔAΔZ, self.H[l])

#             # TODO: for the love of christ I only need the following 2 lines of code to fix this
#             #       I give up. for now...
#             nabla_H = [iconv(ΔAΔZ, self.A[l])] + nabla_H
#             δC_δAlp1 = iconv(ΔAΔZ, self.H[l])

#             # nabla_H = [signal.convolve2d(ΔAΔZ, self.A[l],mode = 'valid')] + nabla_H
#             # δC_δAlp1 = signal.convolve2d(ΔAΔZ, self.H[l],mode = 'full')

#             if self.verbose:
#                 print('******')
#                 print('ΔAΔZ.shape:        ', ΔAΔZ.shape)
#                 print('δC_δAlp1.shape:    ', δC_δAlp1.shape)
#                 print('self.A[l].shape:   ', self.A[l].shape)
#                 print('******')
#                 print()

#         self.feedbackCount = self.feedbackCount + 1

#         for l in range(len(nabla_H)):
#             # print('nabla_H[l].shape:', nabla_H[l].shape)
#             # print('self.nabla_H[l].shape:', self.nabla_H[l].shape)
#             self.nabla_H[l] = self.nabla_H[l] + nabla_H[l]

#         # apply changes if batchsize has been reached
#         if (self.feedbackCount >= self.batchSize) and applyChanges:
            
#             for l in range(len(self.nabla_H)):
#                 if (self.learningFocus == l) or (not self.progressiveLearning):
#                     self.H[l] = self.H[l] + self.nabla_H[l]*self.learningRate/self.feedbackCount
#             self.learningFocus = (self.learningFocus+1)%len(self.nabla_H)
#             self.nabla_H = [H*0 for H in self.H]
#             self.feedbackCount = 0
        
#         return δC_δAlp1

#     def flush(self):
#         self.nabla_H = [H*0 for H in self.H]
#         self.A = []
        
#     def plot(self):

#         f, axarr = plt.subplots(len(self.A),4)
#         for i in range(len(self.H)):


#             axarr[i, 0].imshow(self.A[i], interpolation='none', cmap='Greys')
#             axarr[i, 0].set_title('A['+str(i)+']')
#             axarr[i, 1].imshow(self.Z[i], interpolation='none', cmap='Greys')
#             axarr[i, 1].set_title('Z['+str(i)+']')
#             axarr[i, 2].imshow(self.H[i], interpolation='none', cmap='Greys')
#             axarr[i, 2].set_title('H['+str(i)+']')
#             axarr[i, 3].imshow(self.nabla_H[i], interpolation='none', cmap='Greys')
#             axarr[i, 3].set_title('ΔH['+str(i)+']')
            
#         axarr[-1, 0].imshow(self.A[i+1], interpolation='none', cmap='Greys')
#         axarr[-1, 0].set_title('A['+str(i+1)+']')
        
#         plt.show()


# %%
# (1.7) Teleporter base class and in/out
    # - in training using this module, the system must make a backwards pass without changes on iteration n+1, and then may make changes on iteration n 

class Teleporter_IN(MLModule):

    # initialise class
    def __init__(self, linkedTeleporter, fromStr = '', comment = '', logLevel = logging.WARNING):

        self.linkedTeleporter = linkedTeleporter

        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)

    def output(self):
        return np.array([])

    def feedForward(self):
        return self.output()

    def push(self, input):
        return self.linkedTeleporter.push(input)

    def acceptFeedback(self, feedback):
        return self.linkedTeleporter.ΔX

    def acceptNabla(self, nabla):
        return self.linkedTeleporter.ΔX
    
    def flush(self):
        self.linkedTeleporter.flush()
        return self

class Teleporter_OUT(MLModule):

    # initialise class
    def __init__(self, linkedTeleporter, fromStr = '', comment = '', logLevel = logging.WARNING):

        self.linkedTeleporter = linkedTeleporter

        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)

    def output(self):
        return self.linkedTeleporter.output()

    def feedForward(self):
        return self.output()

    def input(self, input):
        return np.array([])

    def push(self, input):
        return self.output()

    def acceptFeedback(self, feedback):
        return self.linkedTeleporter.acceptFeedback(feedback)

    def acceptNabla(self, nabla):
        return self.linkedTeleporter.acceptNabla(nabla)
    
    def flush(self):
        self.linkedTeleporter.flush()
        return self

class Teleporter(MLModule):

    # initialise class
    def __init__(self, size, fromStr = '', comment = '', logLevel = logging.WARNING, cost_derivative = difference):

        self.IN = Teleporter_IN(self, fromStr = fromStr, comment = comment, logLevel = logLevel)
        self.OUT = Teleporter_OUT(self, fromStr = fromStr, comment = comment, logLevel = logLevel)
        
        self.X = np.zeros(size)
        self.ΔX = np.zeros(size)
        self.cost_derivative = cost_derivative

        super().__init__(fromStr = fromStr, comment = comment, logLevel = logLevel)

    def output(self):
        return self.X

    def feedForward(self):
        return self.output()

    def input(self, input):
        if type(input) != type(None):
            self.X = input
        return self.X

    def push(self, input):
        return self.input(input)

    def acceptFeedback(self, feedback):
        self.ΔX = self.cost_derivative(self.X, feedback)
        return self.ΔX

    def acceptNabla(self, nabla):
        self.ΔX = nabla
        return self.ΔX
    
    def flush(self):
        self.X = self.X*0
        self.ΔX = self.ΔX*0
        return self


# %%
# (2.0) Empirical model class

class EmpiricalModel(basicObj):
    # TODO: add converge modes
    # TODO: add logarithmic scales to steps

    def __init__(self, params, scoreCriteria, mode = 'step'):
        # mode:
            # - 'step':     step across all values between upper and lower bounds
            # - 'converge': test upper, lower, and mid bound. adjust mid bound. re-test mid bound 'steps' times
        self.mode = mode

        # vars:
        # { 'var1Name': {
            # 'upperBound':     float ...,
            # 'lowerBound':     float ..., 
            # 'steps':          float ...,  
            # 'mode':           'linear'/'log'
        # } }
        
        # add in:
        # 'currentStep':    float ..., 
        # 'currentValue':   float ...,
        
        self.params = params
        for p in self.params.values():
            p['currentValue'] = 0

        # scoreCriteria:
        # ['c1', 'c2', ...]
        self.scoreIndices = {}
        for i, c in enumerate(scoreCriteria):
            self.scoreIndices[c] = i

        
        # as many dimensions as there are parameters + 1 for number of scores
        modelShape = (v['steps'] for v in self.params.values())
        self.scoreMatrix = np.zeros((*modelShape, len(scoreCriteria)))

    def __getitem__(self, key):
        return self.params[key]['currentValue']

    def pushScore(self, scores):
        print('self.allPos[self.i]: ',self.allPos[self.i])
        print('self.scoreMatrix.shape: ',self.scoreMatrix.shape)
        for c, v in scores.items():
            print('c,v:', c,v)
            print('index:',(*self.allPos[self.i],self.scoreIndices[c]))
            self.scoreMatrix[(*self.allPos[self.i],self.scoreIndices[c])] = v
            
    def __iter__(self):
        self.allPos = list(np.ndindex(self.scoreMatrix.shape[:-1]))
        self.i = -1
        return self

    def __next__(self):
        self.i=self.i+1
        if self.i >= len(self.allPos):
            for i, step in enumerate(self.getBestPos()):
                variableNames = list(self.params.keys())
                variableName = variableNames[i]
                # TODO: currently scale is linear, do for logarithmic too, and for convergence mode
                self[variableName] = (step*(self.params[variableName]['upperBound']-self.params[variableName]['lowerBound'])/(self.params[variableName]['steps']-1))+(self.params[variableName]['lowerBound'])
            del self.allPos, self.i
            raise StopIteration
        else:
            variableNames = list(self.params.keys())
            currentPos = self.allPos[self.i]
            for i, step in enumerate(currentPos):
                variableName = variableNames[i]
                # TODO: currently scale is linear, do for logarithmic too, and for convergence mode
                self[variableName] = (step*(self.params[variableName]['upperBound']-self.params[variableName]['lowerBound'])/(self.params[variableName]['steps']-1))+(self.params[variableName]['lowerBound'])

            
            return self

    def getBestPos(self):
        combinedScores = np.prod(self.scoreMatrix, axis=-1)
        return tuple(np.argwhere(combinedScores == combinedScores.max())[0])


# # %%
# # (2.1) Empirical function class

# class EmpiricalFunction(basicObj):
#     # TODO: test

#     def __init__(self, steps, XupperBound, XlowerBound, YupperBound, YlowerBound, scoreCriteria, mode = 'step'):
#         # mode:
#             # - 'step':     step across all values between upper and lower bounds
#             # - 'converge': test upper, lower, and mid bound. adjust mid bound. re-test mid bound 'steps' times
#         self.mode = mode
#         self.steps = steps
#         self.XupperBound = XupperBound
#         self.XlowerBound = XlowerBound
#         self.YupperBound = YupperBound
#         self.YlowerBound = YlowerBound
#         self.Xstep = (self.XupperBound - self.XlowerBound)/self.steps
#         self.Ystep = (self.YupperBound - self.YlowerBound)/self.steps
#         # vars:
#         # { 'var1Name': {
#             # 'upperBound':     float ...,
#             # 'lowerBound':     float ..., 
#             # 'steps':          float ...,  
#             # 'mode':           'linear'/'log'
#         # } }
        
#         # add in:
#         # 'currentStep':    float ..., 
#         # 'currentValue':   float ...,
        
#         self.params = np.ones((steps))*self.YlowerBound

#         # scoreCriteria:
#         # ['c1', 'c2', ...]
#         self.scoreIndices = {}
#         for i, c in enumerate(scoreCriteria):
#             self.scoreIndices[c] = i

#     def __call__(self, x):
#         index = int((x-self.XlowerBound)/self.Xstep)
        
#         # if index<0:
#         #     return self.params[0]
#         # elif index>=self.steps:
#         #     return self.params[-1]
#         # else:
#         #     return self.params[index]

#         # conditionless version of code spelled out above
#         return (index<0)*self.params[0] + (index>=self.steps)*self.params[-1] + (index>=0 and index<self.steps)*self.params[index]

#     def pushScore(self, scores):
        
#         for c, v in scores.items():
            
#             self.scoreMatrix[self.i, c] = v
            
#     def __iter__(self):
#         perms = list(np.ndindex((self.steps for p in self.params)))
#         self.allParamPermutations = (np.array(perms)*self.Ystep) + self.YlowerBound
#         self.i = -1
#         self.scoreMatrix = np.zeros(len(perms), len(self.scoreIndices))
#         return self

#     def __next__(self):
#         self.i=self.i+1
        
#         if self.i >= len(self.allParamPermutations):
#             # exit condition
#             self.updateToBest()
#             del self.allParamPermutations, self.i, self.scoreMatrix
#             raise StopIteration

#         else:
#             # normal iteration

#             self.params = self.allParamPermutations[self.i]
#             return self

#     def updateToBest(self):
#         bestIndex = np.argmax(np.prod(self.scoreMatrix, axis=-1))
#         self.params = self.allParamPermutations[bestIndex]
#         return self





