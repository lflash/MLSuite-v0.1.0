# %%
from MLLib import *
import unittest
import random
from scipy import signal

# TODO: - add asserts and logging in each function
#       - remove image/Γ inputs to each function

class TEST_MLLib_Functions(unittest.TestCase):

    def TEST_zero_pad(self):

        testArr = np.ones((2,2))
        print(testArr)
        tl = zero_pad(testArr,(2,2),corner = 'tl')
        bl = zero_pad(testArr,(2,2),corner = 'bl')
        tr = zero_pad(testArr,(2,2),corner = 'tr')
        br = zero_pad(testArr,(2,2),corner = 'br')
        ce = zero_pad(zero_pad(testArr,(3,3),padToShape=True,corner = 'br'),(1,1),corner = 'tl')

        print('tl: ')
        print(tl)
        print('bl: ')
        print(bl)
        print('tr: ')
        print(tr)
        print('br: ')
        print(br)
        print('ce: ')
        print(ce)

        tl = zero_strip(ce,(1,1),corner = 'tl')
        bl = zero_strip(ce,(2,2),corner = 'bl')
        tr = zero_strip(ce,(3,3),corner = 'tr')
        br = zero_strip(ce,(2,3),stripToShape=True,corner = 'br')
        ce = zero_strip(zero_strip(ce,(1,1),corner = 'br'),(1,1),corner = 'tl')

        print('tl: ')
        print(tl)
        print('bl: ')
        print(bl)
        print('tr: ')
        print(tr)
        print('br: ')
        print(br)
        print('ce: ')
        print(ce)

    def TEST_convolution(self, image):

        X_ideal = image
        H_ideal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])+0.00001
        Y_ideal = signal.convolve2d(X_ideal,H_ideal,mode='valid')

        Y = conv(X_ideal,H_ideal,mode='valid')
        X = iconv(Y_ideal,H_ideal,mode='valid')
        H = iconv(Y_ideal,X_ideal,mode='valid')

        print('valid:')
        f, axarr = plt.subplots(3,3)
        print('Y.shape:',Y.shape)
        print('Y_ideal.shape:',Y_ideal.shape)
        axarr[0,0].imshow(Y, interpolation='none', cmap='Greys')
        axarr[0,0].set_title('Y')
        axarr[0,1].imshow(Y_ideal, interpolation='none', cmap='Greys')
        axarr[0,1].set_title('Y_ideal')
        axarr[0,2].imshow(np.around(Y-Y_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[0,2].set_title('Y-Y_ideal')

        print('X.shape:',iconv(Y_ideal,H_ideal).shape)
        print('X_ideal.shape:',X_ideal.shape)
        axarr[1,0].imshow(X, interpolation='none', cmap='Greys')
        axarr[1,0].set_title('iconv(Y,H_ideal)')
        axarr[1,1].imshow(X_ideal, interpolation='none', cmap='Greys')
        axarr[1,1].set_title('X_ideal')
        axarr[1,2].imshow(np.around(X-X_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[1,2].set_title('X-X_ideal')

        print('H.shape:',iconv(Y_ideal,X_ideal).shape)
        print('H_ideal.shape:',H_ideal.shape)
        axarr[2,0].imshow(H, interpolation='none', cmap='Greys')
        axarr[2,0].set_title('iconv(Y,X_ideal)')
        axarr[2,1].imshow(H_ideal, interpolation='none', cmap='Greys')
        axarr[2,1].set_title('H_ideal')
        axarr[2,2].imshow(np.around(H-H_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[2,2].set_title('H-H_ideal')
        plt.show()

        Y_ideal = signal.convolve2d(X_ideal,H_ideal,mode='full')

        Y = conv(X_ideal,H_ideal,mode='full')
        X = iconv(Y_ideal,H_ideal,mode='full')
        H = iconv(Y_ideal,X_ideal,mode='full')

        print('full')
        f, axarr = plt.subplots(3,3)
        print('Y.shape:',Y.shape)
        print('Y_ideal.shape:',Y_ideal.shape)
        axarr[0,0].imshow(Y, interpolation='none', cmap='Greys')
        axarr[0,0].set_title('Y')
        axarr[0,1].imshow(Y_ideal, interpolation='none', cmap='Greys')
        axarr[0,1].set_title('Y_ideal')
        axarr[0,2].imshow(np.around(Y-Y_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[0,2].set_title('Y-Y_ideal')

        print('X.shape:',X.shape)
        print('X_ideal.shape:',X_ideal.shape)
        axarr[1,0].imshow(X, interpolation='none', cmap='Greys')
        axarr[1,0].set_title('iconv(Y,H_ideal)')
        axarr[1,1].imshow(X_ideal, interpolation='none', cmap='Greys')
        axarr[1,1].set_title('X_ideal')
        axarr[1,2].imshow(np.around(X-X_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[1,2].set_title('X-X_ideal')

        print('H.shape:',H.shape)
        print('H_ideal.shape:',H_ideal.shape)
        axarr[2,0].imshow(H, interpolation='none', cmap='Greys')
        axarr[2,0].set_title('iconv(Y,X_ideal)')
        axarr[2,1].imshow(H_ideal, interpolation='none', cmap='Greys')
        axarr[2,1].set_title('H_ideal')
        axarr[2,2].imshow(np.around(H-H_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[2,2].set_title('H-H_ideal')
        plt.show()

        
        H = np.random.randn(*H_ideal.shape)
        Y_ideal = conv(X_ideal,H_ideal)
        Y = conv(X_ideal,H)

        ΔH_ideal = H_ideal - H
        ΔY_ideal = Y_ideal - Y
        ΔH = iconv(ΔY_ideal,X_ideal)


        f, axarr = plt.subplots(3,3)
        axarr[0,0].imshow(H, interpolation='none', cmap='Greys')
        axarr[0,0].set_title('H')
        axarr[0,1].imshow(H_ideal, interpolation='none', cmap='Greys')
        axarr[0,1].set_title('H_ideal')
        axarr[0,2].imshow(H + ΔH, interpolation='none', cmap='Greys')
        axarr[0,2].set_title('H + ΔH')
        axarr[1,0].imshow(ΔH, interpolation='none', cmap='Greys')
        axarr[1,0].set_title('ΔH')
        axarr[1,1].imshow(ΔH_ideal, interpolation='none', cmap='Greys')
        axarr[1,1].set_title('ΔH_ideal')
        axarr[2,0].imshow(Y, interpolation='none', cmap='Greys')
        axarr[2,0].set_title('Y')
        axarr[2,1].imshow(Y_ideal, interpolation='none', cmap='Greys')
        axarr[2,1].set_title('Y_ideal')
        # print('   H,         H_ideal,    ΔH,        ΔH_ideal:', i)
        plt.show()

    def TEST_zero_pad(self,):

        testArr = np.ones((2,2))
        print(testArr)
        tl = zero_pad(testArr,(2,2),corner = 'tl')
        bl = zero_pad(testArr,(2,2),corner = 'bl')
        tr = zero_pad(testArr,(2,2),corner = 'tr')
        br = zero_pad(testArr,(2,2),corner = 'br')
        ce = zero_pad(zero_pad(testArr,(3,3),padToShape=True,corner = 'br'),(1,1),corner = 'tl')

        print('tl: ')
        print(tl)
        print('bl: ')
        print(bl)
        print('tr: ')
        print(tr)
        print('br: ')
        print(br)
        print('ce: ')
        print(ce)

        tl = zero_strip(ce,(1,1),corner = 'tl')
        bl = zero_strip(ce,(2,2),corner = 'bl')
        tr = zero_strip(ce,(3,3),corner = 'tr')
        br = zero_strip(ce,(2,3),stripToShape=True,corner = 'br')
        ce = zero_strip(zero_strip(ce,(1,1),corner = 'br'),(1,1),corner = 'tl')

        print('tl: ')
        print(tl)
        print('bl: ')
        print(bl)
        print('tr: ')
        print(tr)
        print('br: ')
        print(br)
        print('ce: ')
        print(ce)

    def TEST_convolution(image):

        X_ideal = image
        H_ideal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])+0.00001
        Y_ideal = signal.convolve2d(X_ideal,H_ideal,mode='valid')

        Y = conv(X_ideal,H_ideal,mode='valid')
        X = iconv(Y_ideal,H_ideal,mode='valid')
        H = iconv(Y_ideal,X_ideal,mode='valid')

        print('valid:')
        f, axarr = plt.subplots(3,3)
        print('Y.shape:',Y.shape)
        print('Y_ideal.shape:',Y_ideal.shape)
        axarr[0,0].imshow(Y, interpolation='none', cmap='Greys')
        axarr[0,0].set_title('Y')
        axarr[0,1].imshow(Y_ideal, interpolation='none', cmap='Greys')
        axarr[0,1].set_title('Y_ideal')
        axarr[0,2].imshow(np.around(Y-Y_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[0,2].set_title('Y-Y_ideal')

        print('X.shape:',iconv(Y_ideal,H_ideal).shape)
        print('X_ideal.shape:',X_ideal.shape)
        axarr[1,0].imshow(X, interpolation='none', cmap='Greys')
        axarr[1,0].set_title('iconv(Y,H_ideal)')
        axarr[1,1].imshow(X_ideal, interpolation='none', cmap='Greys')
        axarr[1,1].set_title('X_ideal')
        axarr[1,2].imshow(np.around(X-X_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[1,2].set_title('X-X_ideal')

        print('H.shape:',iconv(Y_ideal,X_ideal).shape)
        print('H_ideal.shape:',H_ideal.shape)
        axarr[2,0].imshow(H, interpolation='none', cmap='Greys')
        axarr[2,0].set_title('iconv(Y,X_ideal)')
        axarr[2,1].imshow(H_ideal, interpolation='none', cmap='Greys')
        axarr[2,1].set_title('H_ideal')
        axarr[2,2].imshow(np.around(H-H_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[2,2].set_title('H-H_ideal')
        plt.show()

        Y_ideal = signal.convolve2d(X_ideal,H_ideal,mode='full')

        Y = conv(X_ideal,H_ideal,mode='full')
        X = iconv(Y_ideal,H_ideal,mode='full')
        H = iconv(Y_ideal,X_ideal,mode='full')

        print('full')
        f, axarr = plt.subplots(3,3)
        print('Y.shape:',Y.shape)
        print('Y_ideal.shape:',Y_ideal.shape)
        axarr[0,0].imshow(Y, interpolation='none', cmap='Greys')
        axarr[0,0].set_title('Y')
        axarr[0,1].imshow(Y_ideal, interpolation='none', cmap='Greys')
        axarr[0,1].set_title('Y_ideal')
        axarr[0,2].imshow(np.around(Y-Y_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[0,2].set_title('Y-Y_ideal')

        print('X.shape:',X.shape)
        print('X_ideal.shape:',X_ideal.shape)
        axarr[1,0].imshow(X, interpolation='none', cmap='Greys')
        axarr[1,0].set_title('iconv(Y,H_ideal)')
        axarr[1,1].imshow(X_ideal, interpolation='none', cmap='Greys')
        axarr[1,1].set_title('X_ideal')
        axarr[1,2].imshow(np.around(X-X_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[1,2].set_title('X-X_ideal')

        print('H.shape:',H.shape)
        print('H_ideal.shape:',H_ideal.shape)
        axarr[2,0].imshow(H, interpolation='none', cmap='Greys')
        axarr[2,0].set_title('iconv(Y,X_ideal)')
        axarr[2,1].imshow(H_ideal, interpolation='none', cmap='Greys')
        axarr[2,1].set_title('H_ideal')
        axarr[2,2].imshow(np.around(H-H_ideal,decimals=8), interpolation='none', cmap='Greys')
        axarr[2,2].set_title('H-H_ideal')
        plt.show()

        
        H = np.random.randn(*H_ideal.shape)
        Y_ideal = conv(X_ideal,H_ideal)
        Y = conv(X_ideal,H)

        ΔH_ideal = H_ideal - H
        ΔY_ideal = Y_ideal - Y
        ΔH = iconv(ΔY_ideal,X_ideal)


        f, axarr = plt.subplots(3,3)
        axarr[0,0].imshow(H, interpolation='none', cmap='Greys')
        axarr[0,0].set_title('H')
        axarr[0,1].imshow(H_ideal, interpolation='none', cmap='Greys')
        axarr[0,1].set_title('H_ideal')
        axarr[0,2].imshow(H + ΔH, interpolation='none', cmap='Greys')
        axarr[0,2].set_title('H + ΔH')
        axarr[1,0].imshow(ΔH, interpolation='none', cmap='Greys')
        axarr[1,0].set_title('ΔH')
        axarr[1,1].imshow(ΔH_ideal, interpolation='none', cmap='Greys')
        axarr[1,1].set_title('ΔH_ideal')
        axarr[2,0].imshow(Y, interpolation='none', cmap='Greys')
        axarr[2,0].set_title('Y')
        axarr[2,1].imshow(Y_ideal, interpolation='none', cmap='Greys')
        axarr[2,1].set_title('Y_ideal')
        # print('   H,         H_ideal,    ΔH,        ΔH_ideal:', i)
        plt.show()

    def TEST_activation(Γ,Γ_prime):
        
        # generate values, test if Γ_prime is derivative of Γ
        t = 0.001
        Z = np.arange(-1,2,t)
        Z_prime = np.arange(-1 + t/2, 2 - t/2, t)
        Z = np.dot(np.ones((4,1)),np.reshape(Z,(1,len(Z))))
        Z_prime = np.dot(np.ones((4,1)),np.reshape(Z_prime,(1,len(Z_prime))))
        print(Z)
        print(Z[:,0].reshape(4,1))

        A = np.array([Γ(Z[:,i]) for i in range(len(Z[0,:]))]).transpose()
        print(A)
        A_prime = np.array([Γ_prime(Z_prime[:,i]) for i in range(len(Z_prime[0,:]))]).transpose()
        print(A_prime)
        A_prime_test = np.array([(A[:,i+1]-A[:,i])/t for i in range(len(A_prime[0,:]))]).transpose()
        print(A_prime_test)
        print(np.max(A_prime-A_prime_test))

        # *****************************************************************************************
        print('*****************************************************************************************')
        print('Shallow test:')
        # test that activation can converge to some constant value for constant input
        sizes = [2,3]

        Z = [np.zeros((s, 1)) for s in sizes]
        A = [z*0 for z in Z]
        W = [np.random.randn(sizes[1],sizes[0])]

        # Z_ideal = [np.zeros((s, 1)) for s in sizes]
        # A_ideal = [z*0 for z in Z]
        # W_ideal = [np.random.randn(sizes[i+1],sizes[i]) for i in range(len(sizes[:-1]))]

        ΔZ = [z*0 for z in Z]
        ΔA = [a*0 for a in A]
        ΔW = [w*0 for w in W]
        # ΔW_ideal = [wi - w for wi,w in zip(Z_ideal,Z)]
        
        X = np.random.randn(sizes[0],1)

        Z[0] = X
        A[0] = Γ(Z[0])
        for i in range(500):

            Z[1] = np.dot(W[0],A[0])
            
            Y_ideal = np.ones((sizes[1],1))

            ΔZ[1] = Y_ideal - Z[1]

            ΔW[0] = np.dot(ΔZ[1],A[0].transpose())

            # print('Z[1]:')
            # print(Z[1])
            # print('ΔZ[1]:')
            # print(ΔZ[1])
            # print('ΔW:')
            # print(ΔW)
            # print('W:')
            # print(W)

            W[0] = W[0]+ΔW[0]*0.01
            
        
        print('Z[1]:')
        print(Z[1])
        print('ΔZ[1]:')
        print(ΔZ[1])
        print('ΔW:')
        print(ΔW)
        print('W:')
        print(W)

        # *****************************************************************************************
        print('*****************************************************************************************')
        print('Deep test 1:')
        sizes = [1,200,1]

        Z = [np.zeros((s, 1)) for s in sizes]
        A = [z*0 for z in Z]
        W = [np.random.randn(sizes[i+1],sizes[i]) for i in range(len(sizes)-1)]

        ΔZ = [z*0 for z in Z]
        # ΔA = [a*0 for a in A]
        ΔW = [w*0 for w in W]

        
        for i in range(100):
            
            X = np.random.randn(sizes[0],1)
            Y_ideal = -X/2

            Z[0] = X
            A[0] = Γ(Z[0])
            Z[1] = np.dot(W[0],A[0])
            A[1] = Γ(Z[1])
            Z[2] = np.dot(W[1],A[1])

            ΔZ[2] = Y_ideal - Z[2]

            ######################################

            ΔW[1] = np.dot(ΔZ[2],A[1].transpose())

            ΔZ[1] = np.dot(np.transpose(W[1]),ΔZ[2])*Γ_prime(Z[1])

            ######################################

            ΔW[0] = np.dot(ΔZ[1],A[0].transpose())

            ΔZ[0] = np.dot(np.transpose(W[0]),ΔZ[1])*Γ_prime(Z[0])

            ######################################

            W[1] = W[1]+ΔW[1]*0.01
            W[0] = W[0]+ΔW[0]*0.01


        print('*****************************************************************************************')
        print('Deep test 2:')
        sizes = [1,200,1]

        Z = [np.zeros((s, 1)) for s in sizes]
        A = [z*0 for z in Z]
        W = [np.random.randn(sizes[i+1],sizes[i]) for i in range(len(sizes)-1)]

        # ΔA = [a*0 for a in A]
        ΔZ = [z*0 for z in Z]
        ΔW = [w*0 for w in W]
        ΔW_batch = [w*0 for w in W]

        batchSize = 10
        learningRate = 0.01
        
        for i in range(100):

            
            
            X = np.random.randn(sizes[0],1)
            Y_ideal = -X/2

            Z[0] = X
            A[0] = Γ(Z[0])
            Z[1] = np.dot(W[0],A[0])
            A[1] = Γ(Z[1])
            Z[2] = np.dot(W[1],A[1])

            ΔZ[2] = Y_ideal - Z[2]

            ######################################

            for l in list(range(len(sizes)))[::-1][:-1]:

                ΔW[l-1] = np.dot(ΔZ[l],A[l-1].transpose())

                ΔZ[l-1] = np.dot(np.transpose(W[l-1]),ΔZ[l])*Γ_prime(Z[l-1])


            ΔW_batch = [δwb + δw for δwb, δw in zip(ΔW_batch,ΔW)] 

            for l in range(len(sizes)-1):
                W[l] = W[l]+ΔW[l]*0.01

            
        
        
        print('Z:')
        print(Z)
        print('ΔZ:')
        print(ΔZ)
        print('ΔW:')
        print(ΔW)
        print('W:')
        print(W)
        print('Y_ideal:')
        print(Y_ideal)
        print('ΔZ[-1]:')
        print(ΔZ[-1])
        print('Z[-1]:')
        print(Z[-1])

    def TEST_zipFuncs(self):
        Γ = zipFuncs([sigmoid, relu])
        X = np.random.randn(6,1)
        print('sigmoid:\n', sigmoid(X))
        print('relu:\n', relu(X))
        print('Γ:\n', Γ(X))

# TODO: - include all functions in NN
class TEST_NN(unittest.TestCase):
    def TEST_NN_basicFunctions(self):
        def shallowTest(batchSize, learningRate, activation, activation_prime):
            print('activation: ', activation.__name__)
            testNet = NN(layerSizes = [2,3,4,2], batchSize = batchSize, learningRate = learningRate, activation = activation, activation_prime = activation_prime)
            cost = 0
            for i in range(100):
                X = np.random.rand(2,1)
                Y = testNet.push(X)
                Y_ideal = X*np.array([[0.5], [-1]]) + np.array([[0.1], [1]])
                cost = cost + np.sum((Y-Y_ideal)**2)
             
            print('\tcost 0:', cost/100)
                
            for i in range(1000):
                X = np.random.rand(2,1)
                Y_ideal = X*np.array([[0.5], [-1]]) + np.array([[0.1], [1]])
                Y = testNet.push(X)
                testNet.acceptFeedback(Y_ideal)
            
            cost = 0
            for i in range(100):
                X = np.random.rand(2,1)
                Y = testNet.push(X)
                Y_ideal = X*np.array([[0.5], [-1]]) + np.array([[0.1], [1]])
                cost = cost + np.sum((Y-Y_ideal)**2)
            print('\tcost 1:', cost/100)

        shallowTest(10, 0.05, sigmoid, sigmoid_prime)
        shallowTest(10, 0.05, swish, swish_prime)
        shallowTest(10, 0.05, relu, relu_prime)
        shallowTest(10, 0.05, sins, sins_prime)


    def TEST_NN_deepTest(self):

        # initialise NN and data
        testNN = NN(layerSizes = [784,1000,1000,10],learningRate = 0.2,activation = swish, activation_prime = swish_prime, batchSize = 50, logLevel = logging.DEBUG)
        (trainingData,testData) = get_training_data()
        
        # initial test
        testScore = 0
        for d in testData:
            testScore = testScore + int(d['Y'] == np.argmax(testNN.push(d['X'])))
        testScore = testScore/len(testData)
        print('testScore: ',round(testScore*100,2),'%')

        # one training epoch
        random.shuffle(trainingData)
        for i, d in enumerate(trainingData):
            testNN.push(d['X'])
            testNN.acceptFeedback(numToArray(d['Y']))
            if i%1000 == 0:
                print('i: ',i,'/',len(trainingData),end="\r")

        # re-test
        testScore = 0
        for d in testData:
            testScore = testScore + int(d['Y'] == np.argmax(testNN.push(d['X'])))
        testScore = testScore/len(testData)
        print('testScore: ',round(testScore*100,2),'%')
        
# TODO: - include all functions in ZNN
class TEST_ZNN(unittest.TestCase):
    def TEST_ZNN_basicFunctions(self):
        testNet = ZNN(layerSizes = [2,3,4,2], batchSize = 2, learningRate = 0.3)
        print(testNet)
        Y = testNet.push(np.array([1,1]))
        testNet.acceptFeedback(np.array([1,1]))
        print(testNet)

    def TEST_ZNN_deepTest(self):

        # initialise ZNN and data
        testZNN = ZNN(layerSizes = [784,100,100,10],learningRate = 0.05,activation = sigmoid, activation_prime = sigmoid, batchSize = 50, logLevel = logging.DEBUG)
        (trainingData,testData) = get_training_data()
        
        # initial test
        testScore = 0
        for d in testData:
            testScore = testScore + int(d['Y'] == np.argmax(testZNN.push(d['X'])))
        testScore = testScore/len(testData)
        print('testScore: ',round(testScore*100,2),'%')

        # one training epoch
        random.shuffle(trainingData)
        for i, d in enumerate(trainingData):
            testZNN.push(d['X'])
            testZNN.acceptFeedback(numToArray(d['Y']))
            if i%1000 == 0:
                print('i: ',i,'/',len(trainingData),end="\r")

        # re-test
        testScore = 0
        for d in testData:
            testScore = testScore + int(d['Y'] == np.argmax(testZNN.push(d['X'])))
        testScore = testScore/len(testData)
        print('testScore: ',round(testScore*100,2),'%')


class TEST_FL(unittest.TestCase):
    def TEST_FL_basicFunctions(self):
        testFL = FL(sigmoid, sigmoid_prime)
        X1 = np.random.rand(5)
        Y1 = sigmoid(X1)
        Y_ideal1 = np.random.rand(5)
        ΔX1 = testFL.cost_derivative(Y1, Y_ideal1)


        Y2 = testFL.push(X1)
        assert Y2.all() == Y1.all()

        ΔX2 = testFL.acceptFeedback(Y_ideal1)
        assert ΔX2.all() == ΔX1.all()
   

class TEST_ModuleSeries(unittest.TestCase):
    def TEST_ModuleSeries_init(self):
        # initialise NN and data
        N1 = NN(layerSizes = [784,100],learningRate = 0.05, activation = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)
        N2 = NN(layerSizes = [100,100],learningRate = 0.05, activation = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)
        N3 = NN(layerSizes = [100,10],learningRate = 0.05, activation = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)
        testNN = ModuleSeries([N1, N2, N3])

        (trainingData,testData) = get_training_data()
        
        # initial test
        testScore = 0
        for d in testData:
            testScore = testScore + int(d['Y'] == np.argmax(testNN.push(d['X'])))
        testScore = testScore/len(testData)
        print('testScore: ',round(testScore*100,2),'%')

        # one training epoch
        random.shuffle(trainingData)
        for i, d in enumerate(trainingData):
            testNN.push(d['X'])
            testNN.acceptFeedback(numToArray(d['Y']))
            if i%1000 == 0:
                print('i: ',i,'/',len(trainingData),end="\r")

        # re-test
        testScore = 0
        for d in testData:
            testScore = testScore + int(d['Y'] == np.argmax(testNN.push(d['X'])))
        testScore = testScore/len(testData)
        print('testScore: ',round(testScore*100,2),'%')

    # def TEST_ModuleSeries_push(self):
    #     pass

    # def TEST_ModuleSeries_backprop(self):
    #     pass

class TEST_ModuleParallel(unittest.TestCase):
    def TEST_ModuleParallel_init(self):
        # initialise NN and data
        N1 = NN(layerSizes = [784,101],learningRate = 0.05, activation = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)
        N2 = NN(layerSizes = [100,100],learningRate = 0.05, activation = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)
        # N3 = NN(layerSizes = [100,100],learningRate = 0.05, activation = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)
        # N4 = NN(layerSizes = [100,10],learningRate = 0.05, activation = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)

        # testNN = ModuleSeries([N1, N2, N4])

        N3 = NN(layerSizes = [1,1],learningRate = 0.05, activation = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)
        N4 = NN(layerSizes = [101,10],learningRate = 0.05, activation 
        = sigmoid, activation_prime = sigmoid_prime, batchSize = 50, logLevel = logging.DEBUG)

        testNN = ModuleSeries([N1, [N2, N3], N4])

        (trainingData,testData) = get_training_data()

        testNN.logger.setLevel(logging.WARNING)
        # initial test
        testScore = 0
        for d in testData:
            testScore = testScore + int(d['Y'] == np.argmax(testNN.push(d['X'])))
            #break
        testScore = testScore/len(testData)
        print('testScore: ',round(testScore*100,2),'%')

        testNN.logger.setLevel(logging.WARNING)

        # one training epoch
        random.shuffle(trainingData)
        for i, d in enumerate(trainingData):
            testNN.push(d['X'])
            testNN.acceptFeedback(numToArray(d['Y']))
            if i%1000 == 0:
                print('i: ',i,'/',len(trainingData),end="\r")
            #break
        testNN.logger.setLevel(logging.WARNING)
        # re-test
        testScore = 0
        for d in testData:
            testScore = testScore + int(d['Y'] == np.argmax(testNN.push(d['X'])))
            #break
        testScore = testScore/len(testData)
        print('testScore: ',round(testScore*100,2),'%')

class TEST_Teleporter(unittest.TestCase):
    def TEST_Teleporter_init(self):
        tt = Teleporter((5,1))

        t_in = tt.IN
        t_out = tt.OUT
        X1 = np.random.randn(5)
        X2 = np.random.randn(5)
        X3 = np.random.randn(5)
        X4 = np.random.randn(5)
        Y1 = np.random.randn(5)
        Y2 = np.random.randn(5)
        Y3 = np.random.randn(5)
        Y4 = np.random.randn(5)
        print('X1: ', X1)
        print('X2: ', X2)
        print('X3: ', X3)
        print('X4: ', X4)
        print()
        print('t_in.push(X1):  ', t_in.push(X1))
        assert tt.X.all() == X1.all()
        print('t_in.input(X2): ', t_in.input(X2))
        assert tt.X.all() == X2.all()
        print('t_out.push(X1): ', t_out.push(X3))
        assert tt.X.all() == X2.all()
        print('t_out.input(X2):', t_out.input(X4))
        assert tt.X.all() == X2.all()
        print()
        print('Y1: ', Y1)
        print('Y2: ', Y2)
        print('Y3: ', Y3)
        print('Y4: ', Y4)
        print()
        print('t_out.acceptNabla(Y1):   ', t_out.acceptNabla(Y1))
        assert tt.ΔX.all() == tt.cost_derivative(tt.X,Y1).all()
        print('t_out.acceptFeedback(Y2):', t_out.acceptFeedback(Y2))
        assert tt.ΔX.all() == Y2.all()
        print('t_in.acceptFeedback(Y2): ', t_in.acceptFeedback(Y3))
        assert tt.ΔX.all() == Y2.all()
        print('t_in.acceptNabla(Y1):    ', t_in.acceptNabla(Y4))
        assert tt.ΔX.all() == Y2.all()


class TEST_EmpiricalModel(unittest.TestCase):
    def TEST_EmpiricalModel_init(self):
        testVars = { 'v1': {'lowerBound': 2, 'upperBound': 5, 'steps':  3, 'mode': 'linear'}, 'v2': {'lowerBound': -1, 'upperBound': 5, 'steps':  6, 'mode': 'linear'} }
        testModel = EmpiricalModel(testVars, ['c1', 'c2'])
        print('testModel: ', testModel)
        

    def TEST_EmpiricalModel_iterate(self):
        testVars = { 'v1': {'lowerBound': 2, 'upperBound': 5, 'steps':  3, 'mode': 'linear'}, 'v2': {'lowerBound': -1, 'upperBound': 5, 'steps':  6, 'mode': 'linear'} }
        testModel = EmpiricalModel(testVars, ['c1', 'c2'])
        for i, m in enumerate(testModel):
            print('i: ', i)
            print('\tm[\'v1\']: ', m['v1'])
            print('\tm[\'v2\']: ', m['v2'])
            print('\tscores: ',{'c1':i,'c2':m['v2']*i**2})
            print()
            m.pushScore({'c1':i,'c2':-(m['v2']**2)+1})

        print('testModel: ', testModel)
        combinedScores = np.prod(testModel.scoreMatrix, axis=-1)
        print(np.prod(testModel.scoreMatrix, axis=-1))
        print(np.argmax(combinedScores))

        print(tuple(np.argwhere(combinedScores == combinedScores.max())))

        print(testModel.getBestPos())
        print()



# TODO: test the argmax_prime cost derivative
if __name__ == '__main__':
    pass
    # unittest.main()
    # test = TEST_EmpiricalModel()
    # test.TEST_EmpiricalModel_init()
    # test.TEST_EmpiricalModel_iterate()

    # test = TEST_NN()
    # test.TEST_NN_basicFunctions()
    # test.TEST_NN_deepTest()
    test = TEST_ModuleSeries()
    test.TEST_ModuleSeries_init()

# %%
