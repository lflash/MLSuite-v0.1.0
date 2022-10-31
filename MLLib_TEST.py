# %%
from MLLib import *
import unittest
import random

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

# TODO: - include all functions in NN
class TEST_NN(unittest.TestCase):
    def TEST_NN_basicFunctions(self):
        testNet = NN(layerSizes = [2,3,4,2], batchSize = 2, learningRate = 0.3)
        print(testNet)
        Y = testNet.push(np.array([1,1]))
        testNet.acceptFeedback(np.array([1,1]))
        print(testNet)

    def TEST_NN_deepTest(self):

        # initialise NN and data
        testNN = NN(layerSizes = [784,100,100,10],learningRate = 0.05,activation = sigmoid, activation_prime = sigmoid, batchSize = 50, logLevel = logging.DEBUG)
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
        

if __name__ == '__main__':
    # unittest.main()
    test = TEST_NN()
    test.TEST_NN_deepTest()

# %%

"""
# # (2.0) NN basic tests

# # testNet = NN(verbose = False, layerSizes = [2,3,4,2], batchSize = 2, learningRate = 0.3)
# # print(testNet)
# # Y = testNet.push(np.array([1,1]))
# # testNet.acceptFeedback(np.array([1,1]))

# # print(testNet)

# # %% sins test





# # %% ZNN proof of concept


# sizes = [2,4]
# Z = [np.zeros((s, 1)) for s in sizes]
# A = [z*0 for z in Z]
# W = [np.random.randn(sizes[i+1],sizes[i]) for i in range(len(sizes[:-1]))]    # TODO: Neurons should only connect to a small number of other neurons, not all. could fix w/ fimple filter func

# Z_ideal = [np.zeros((s, 1)) for s in sizes]
# A_ideal = [z*0 for z in Z]
# W_ideal = [np.ones((sizes[i+1],sizes[i])) for i in range(len(sizes[:-1]))]    # TODO: Neurons should only connect to a small number of other neurons, not all. could fix w/ fimple filter func


# ΔZ = [z*0 for z in Z]
# ΔA = [a*0 for a in A]
# ΔW = [w*0 for w in W]

# Γ = sins
# Γ_prime = sins_prime
# score = 0
# for i in range(1):

#     # init IO

#     X = np.random.randn(2,1)
    
#     Z = [z*0 for z in Z]
#     A = [a*0 for a in A]
#     ΔZ = [z*0 for z in Z]
#     ΔA = [a*0 for a in A]
#     ΔW = [w*0 for w in W]


#     # feed forward

#     A[0] = X
#     A_ideal[0] = X
#     for l in range(len(Z)-1):
            
#         Z[l+1] = np.dot(W[l], A[l])
#         A[l+1] = sins(Z[l+1])
#         Z_ideal[l+1] = np.dot(W_ideal[l], A_ideal[l])
#         A_ideal[l+1] = sins(Z_ideal[l+1])
#     Y = Z_ideal[-1]
#     score = score*0.99 + 0.01*(sum(abs(Y-Z[-1]))/sum(abs(Y)))
#     print('i, score: ', i, score)
#     # print('**************************************************')
#     # print('X:  ',X.shape,'\n', X)
#     # print('Y:  ',Y.shape,'\n', Y)
#     # print('Z:  ',[z.shape for z in Z],'\n', Z)
#     # print('ΔZ: ',[z.shape for z in ΔZ],'\n', ΔZ)
#     # print('A:  ',[z.shape for z in A],'\n', A)
#     # print('ΔA: ',[z.shape for z in ΔA],'\n', ΔA)
#     # print('W:  ',[z.shape for z in W],'\n', W)
#     # print('ΔW: ',[z.shape for z in ΔW],'\n', ΔW)
#     # backprop

#     ΔZ[-1] = Y-Z[-1]
    
#     for l in list(range(len(Z)))[::-2]:

        
#         # δC/δW[l-1]    =   δC/δZ[l]        δZ[l]/δW[l-1]
#         ΔW[l-1]         =   np.dot(ΔZ[l], A[l-1].transpose())

#         # δC/δA[l-1]    =   δC/δZ[l]        δZ[l]/δA[l-1]
#         ΔA[l-1]         =   np.dot(np.transpose(W[l-1]), ΔZ[l])

#         # δC/δZ[l-1]    =   δC/δA[l-1]      δA[l-1]/δZ[l-1]
#         ΔZ[l-1]         =   ΔA[l-1]*sins_prime(Z[l-1])

#     # for i, δW in enumerate(ΔW):
#     #     W[i] = W[i]+δW*0.05


# print('**************************************************')
# print('X:\n', X)
# print('Y:\n', Y)
# print('Z:\n', Z)
# print('ΔZ:\n', ΔZ)
# print('Z_ideal:\n', Z_ideal)
# print('A:\n', A)
# print('ΔA:\n', ΔA)
# print('A_ideal:\n', A_ideal)
# print('W:\n', W)
# print('ΔW:\n', ΔW)
# print('W_ideal:\n', W_ideal)


# # %%
# # (2.1) import and process NN deep training data

# # purposes:
# #   training
# #   test
# #   generated
# # types:
# #   images
# #   digits
# #   inputs



# # plt.imshow(trainingImages[0], interpolation='none', cmap='Greys')
# # plt.show()
# (trainingData,testData) = get_training_data()

# TEST_convolution(trainingData[0]['X'])
# # %%
# # (2.2) shallow test NNs
# def TEST_NN_shallowTest(testNN):
#     Γ = testNN.activation
#     X = [(random.random(),random.random()) for i in range(1000)]
#     trainingData = [{'X':np.array([a,b]),'Y':np.array(Γ(np.array([a+b])))} for a,b in X]
#     score = 1
#     for i, d in enumerate(trainingData):
#         out = testNN.push(d['X'])
#         testNN.acceptFeedback(d['Y'])
#         score = score*0.99 + 0.01*abs((out)-d['Y'])
#         if i%100 == 0:
#             # print('out: ', out)
#             # print('d["Y"]: ', d['Y'])
#             print('i, score: ',i,'/',len(trainingData), score)#,end="\r")
#             #print(testNN)
#     return testNN

# def TEST_ZNN_shallowTest(testNN, verbose = False):
#     Γ = testNN.activation
#     X = [(random.random(),random.random()) for i in range(1000)]
#     trainingData = [{'X':np.array([a,b]),'Y':np.array([a+b])} for a,b in X]
#     score = 1
#     for i, d in enumerate(trainingData):
#         out = testNN.push(d['X'])
#         testNN.acceptFeedback(d['Y'])
#         score = score*0.99 + 0.01*abs((out)-d['Y'])
        
#         if i%100 == 0 and verbose:
#             print('****************************************************************')
#             print('out: ', out)
#             print('d["Y"]: ', d['Y'])
#             print('i, score: ',i,'/',len(trainingData), score)#,end="\r")
#             print(testNN)
#             print('****************************************************************')
#             print()
#         elif i%100 == 0:
#             print('i, score: ',i,'/',len(trainingData), score)#,end="\r")
#     return testNN

# print('sins:')
# testNN = ZNN(layerSizes = [2,10,1],learningRate = 0.03, batchsize = 19, activation = 'sins', activation_prime = 'sins')
# TEST_ZNN_shallowTest(testNN)#, verbose=True)
# print()

# print(testNN)


# # print('sigmoid:')
# # testNN = NN(layerSizes = [2,1],learningRate = 0.3, activation = 'sigmoid', activation_prime = 'sigmoid_prime')
# # TEST_shallowTest(testNN)
# # print()
# # print('swish:')
# # testNN = NN(layerSizes = [2,1],learningRate = 0.3, activation = 'swish', activation_prime = 'swish_prime')
# # TEST_shallowTest(testNN)
# # print()
# # print('linear:')
# # testNN = NN(layerSizes = [2,1],learningRate = 0.3, activation = 'linear', activation_prime = 'linear_prime')
# # TEST_shallowTest(testNN)
# # print()
# # print('relu:')
# # testNN = NN(layerSizes = [2,1],learningRate = 0.3, activation = 'relu', activation_prime = 'relu_prime')
# # TEST_shallowTest(testNN)
# # print()
# # print('sins:')
# # testNN = NN(layerSizes = [2,1],learningRate = 0.01, activation = 'sins', activation_prime = 'sins_prime')
# # TEST_shallowTest(testNN)

# # %% deep test NNs
# def TEST_deepTest(testNN):
#     (trainingData,testData) = get_training_data()
#     random.shuffle(trainingData)
#     for i, d in enumerate(trainingData):
#         testNN.push(d['X'])
#         testNN.acceptFeedback(numToArray(d['Y']))
#         if i%1000 == 0:
#             print('i: ',i,'/',len(trainingData),end="\r")

#     testScore = 0
#     for d in testData:
        
#         testScore = testScore + int(d['Y'] == np.argmax(testNN.push(d['X'])))
#     testScore = testScore/len(testData)

#     print('testScore: ',round(testScore*100,2),'%')
#     return testNN

# testNN = NN(layerSizes = [784,100,100,10],learningRate = 0.05,activation = 'swish', activation_prime = 'swish_prime', batchSize = 50)
# TEST_deepTest(testNN)
# TEST_deepTest(testNN)
# TEST_deepTest(testNN)




# # %%
# # (2.4) test single layer CNN class

# (trainingData,testData) = get_training_data()

# testCNN = CNN(filterSizes = [3], batchSize = 5, learningRate = 0.1, progressiveLearning = False, verbose = False)

# sobelFiltV = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
# sobelFiltH = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
# H_ideal = sobelFiltH
# for i in range(50):


#     X = trainingData[i]['X']
    
#     Y_ideal = conv(X,H_ideal)
#     Y = conv(X,testCNN.H[0])

#     ΔH_ideal = H_ideal - testCNN.H[0]
#     ΔY_ideal = Y_ideal - Y
#     ΔH = iconv(ΔY_ideal,X)





#     X = trainingData[i]['X']
#     Y_ideal = signal.convolve2d(X,H_ideal, mode='valid')
#     Y = testCNN.push(X)
#     testCNN.acceptFeedback(Y_ideal)

#     f, axarr = plt.subplots(3,3)
#     axarr[0,0].imshow(testCNN.H[0], interpolation='none', cmap='Greys')
#     axarr[0,0].set_title('H')
#     axarr[0,1].imshow(H_ideal, interpolation='none', cmap='Greys')
#     axarr[0,1].set_title('H_ideal')
#     axarr[0,2].imshow(testCNN.H[0] + ΔH, interpolation='none', cmap='Greys')
#     axarr[0,2].set_title('H + ΔH')
#     axarr[1,0].imshow(ΔH, interpolation='none', cmap='Greys')
#     axarr[1,0].set_title('ΔH')
#     axarr[1,1].imshow(ΔH_ideal, interpolation='none', cmap='Greys')
#     axarr[1,1].set_title('ΔH_ideal')
#     axarr[1,2].imshow(testCNN.nabla_H[0], interpolation='none', cmap='Greys')
#     axarr[1,2].set_title('testCNN.nabla_H[0]')
#     axarr[2,0].imshow(Y, interpolation='none', cmap='Greys')
#     axarr[2,0].set_title('Y')
#     axarr[2,1].imshow(Y_ideal, interpolation='none', cmap='Greys')
#     axarr[2,1].set_title('Y_ideal')
#     # print('   H,         H_ideal,    ΔH,        ΔH_ideal:', i)
#     plt.show()

#     # for i, a in enumerate(testCNN.A):
#     #     print('i, a.shape: ',i,a.shape)
    
    
#     #testCNN.plot()

# print('testCNN.A[0].shape, testCNN.A[1].shape: ', testCNN.A[0].shape, testCNN.A[1].shape)
# print('testCNN.H[0]: \n', testCNN.H[0])
# print('testCNN.nabla_H[0]: \n', testCNN.nabla_H[0])
# print('ΔH: \n', ΔH)

# print('Y_ideal:')
# plt.imshow(Y_ideal, interpolation='none', cmap='Greys')
# plt.show()
# print('Y:')
# plt.imshow(Y, interpolation='none', cmap='Greys')
# plt.show()


# # %%
# # (2.5) test CNN class
# (trainingData,testData) = get_training_data()

# testCNN = CNN(filterSizes = [3,3], batchSize = 5, learningRate = 0.1, progressiveLearning = False, verbose = False)

# sobelFiltV = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
# sobelFiltH = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
# H_ideal = sobelFiltH
# for i in range(500):
#     X = trainingData[i]['X']
#     Y = testCNN.push(X)
    
    
#     Y_ideal = conv(conv(X,H_ideal),sobelFiltV)
    

#     ΔH_ideal = H_ideal - testCNN.H[0]
#     ΔY_ideal = Y_ideal - Y
#     ΔH = iconv(ΔY_ideal,X)

    
#     testCNN.acceptFeedback(Y_ideal)
#     testCNN.plot()
#     # f, axarr = plt.subplots(3,2)
#     # axarr[0,0].imshow(testCNN.H[-1], interpolation='none', cmap='Greys')
#     # axarr[0,0].set_title('H')
#     # axarr[0,1].imshow(H_ideal, interpolation='none', cmap='Greys')
#     # axarr[0,1].set_title('H_ideal')
#     # # axarr[0,2].imshow(testCNN.H[-1] + ΔH, interpolation='none', cmap='Greys')
#     # # axarr[0,2].set_title('H + ΔH')
#     # axarr[1,0].imshow(ΔH, interpolation='none', cmap='Greys')
#     # axarr[1,0].set_title('ΔH')
#     # axarr[1,1].imshow(ΔH_ideal, interpolation='none', cmap='Greys')
#     # axarr[1,1].set_title('ΔH_ideal')
#     # # axarr[1,2].imshow(testCNN.nabla_H[-1], interpolation='none', cmap='Greys')
#     # # axarr[1,2].set_title('testCNN.nabla_H[0]')
#     # axarr[2,0].imshow(Y, interpolation='none', cmap='Greys')
#     # axarr[2,0].set_title('Y')
#     # axarr[2,1].imshow(Y_ideal, interpolation='none', cmap='Greys')
#     # axarr[2,1].set_title('Y_ideal')
#     # # print('   H,         H_ideal,    ΔH,        ΔH_ideal:', i)
#     # plt.show()

#     # for i, a in enumerate(testCNN.A):
#     #     print('i, a.shape: ',i,a.shape)
    
    
#     #testCNN.plot()
# # %%
# print('testCNN.A[0].shape, testCNN.A[1].shape: ', testCNN.A[0].shape, testCNN.A[1].shape)
# print('testCNN.H[0]: \n', testCNN.H[0])
# print('testCNN.nabla_H[0]: \n', testCNN.nabla_H[0])
# print('ΔH: \n', ΔH)

# print('Y_ideal:')
# plt.imshow(Y_ideal, interpolation='none', cmap='Greys')
# plt.show()
# print('Y:')
# plt.imshow(Y, interpolation='none', cmap='Greys')
# plt.show()

# # %%

# # # %%
# # # NN deep training

# # testNet = NN(verbose = False, layerSizes = [784,200,100,100,10], batchSize = 10, learningRate = 0.5)
# # score = 0
# # prevScore = 0

# # for i in range(len(trainingOutput)):
# #     Y = testNet.push(trainingInputs[i])
# #     testNet.acceptFeedback(trainingOutput[i])
# #     score = score*0.999 + (Y.argmax()==trainingOutput[i].argmax())*0.001
# #     prevScore = score + 0
# #     if i%100 == 0:
        
# #         print(i,'/',len(trainingOutput),'  score: ', round(score,2))

# # # %%
# # # NN layers deep training

# # L1 = NN(verbose = False, layerSizes = [784,200], batchSize = 10, learningRate = 0.5)
# # L2 = NN(verbose = False, layerSizes = [200,100], batchSize = 10, learningRate = 0.5)
# # L3 = NN(verbose = False, layerSizes = [100,100], batchSize = 10, learningRate = 0.5)
# # L4 = NN(verbose = False, layerSizes = [100,10], batchSize = 10, learningRate = 0.5)
# # net = [L1, L2, L3, L4]
# # score = 0
# # prevScore = 0

# # for i in range(len(trainingOutput)):
# #     A = trainingInputs[i]
# #     for l in net:
# #         A = l.push(A)

# #     nabla = net[-1].acceptFeedback(trainingOutput[i])
# #     for l in net[::-1][1:]:
# #         nabla = l.acceptNabla(nabla)
        
    
    
# #     score = score*0.999 + (A.argmax()==trainingOutput[i].argmax())*0.001
# #     prevScore = score + 0
# #     if i%100 == 0:
        
# #         print(i,'/',len(trainingOutput),'  score: ', round(score,2))

# # # %%    
# # # NN deep test
# # score = 0
# # iterator = list(range(len(testOutput)))
# # random.shuffle(iterator)
# # num = 0
# # for i in iterator:
# #     Y = testNet.push(testInputs[i])
    
# #     score = score + (Y.argmax()==testOutput[i].argmax())
# #     if i%100 == 0:
        
# #         print(num,'/',len(testOutput),'  score: ', score)
# #     num = num+1



# # %%
# # GAN shallow test
# generator = NN(verbose = False, layerSizes = [50,784], batchSize = 200, learningRate = 0.5, activation = 'swish', activation_prime = 'swish_prime')
# discriminator = NN(verbose = False, layerSizes = [784,200,100,1], batchSize = 200, learningRate = 0.5, activation = 'swish', activation_prime = 'swish_prime')


# # plt.imshow(trainingInputs[0].reshape(trainingImages[0].shape), interpolation='none', cmap='Greys')
# # plt.show()

# # # %%
# # # Train NN to create image

# # for i in range(1000):
# #     Y = generator.push(np.array([1]))
# #     generator.acceptFeedback(trainingInputs[0])
    

# # plt.imshow(Y.reshape(trainingImages[0].shape), interpolation='none', cmap='Greys')
# # plt.show()

# # %%
# # Train discriminator
# discriminatorInputs = []
# # make trainingData
# for i in range(len(trainingData)-40000):
#     discriminatorInputs = discriminatorInputs + [{'i': sigmoid(np.random.randn(784, 1)),'g':np.array([1])}, {'i':trainingInputs[i],'g':np.array([0])}]
#     print("\r",i,'/',len(trainingData), end="")
    

# random.shuffle(discriminatorInputs)

# score = 0


# for i, d in enumerate(discriminatorInputs):
#     Y = discriminator.push(d['i'])
#     discriminator.acceptFeedback(d['g'])
#     score = score*0.999 + int(((Y[0][0]>0.5)==(d['g'][0]==1)))*0.001
    
#     print("\r",i,'/',len(discriminatorInputs), '  score: ', round(score, 3), end="")

#     if score > 0.999:
#         break



# # %%
# # # Gan shallow training

# # prev_nabla = discriminator.acceptFeedback(np.array([1]))
# # while True:
    
# #     # generate images
# #     generatedImage = generator.push(np.array([1]))
# #     print('image: ')
# #     plt.imshow(generatedImage.reshape(trainingImages[0].shape), interpolation='none', cmap='Greys')
# #     plt.show()
    
# #     # plt.imshow(generatedImage.reshape(trainingImages[0].shape), interpolation='none', cmap='Greys')
# #     # plt.show()

# #     # create discriminator training set
# #     discriminatorInputs = []
# #     for i in range(1000):
# #         discriminatorInputs = discriminatorInputs + [{'i': copy.copy(generatedImage),'g':np.array([random.random()])}, {'i':trainingInputs[0],'g':np.array([0])}]
    
# #     random.shuffle(discriminatorInputs)
# #     # train discriminator
# #     discriminator = NN(verbose = False, layerSizes = [784,1], batchSize = 20, learningRate = 1, activation = 'sigmoid', activation_prime = 'sigmoid_prime')
# #     for d in discriminatorInputs:
# #         discriminator.push(d['i'])
# #         discriminator.acceptFeedback(d['g'])

# #     # train generator
# #     for i in range(1000):
# #         Y = generator.push(np.array([1]))
# #         discriminator.push(Y)
# #         nabla = discriminator.acceptFeedback(np.array([1]), applyChanges=False)
# #         generator.acceptNabla(nabla)

# #     delta_nabla = nabla-prev_nabla
# #     prev_nabla = copy.deepcopy(nabla)
# #     plt.imshow(nabla.reshape(trainingImages[0].shape), interpolation='none', cmap='Greys')
# #     plt.show()
# #     plt.imshow(delta_nabla.reshape(trainingImages[0].shape), interpolation='none', cmap='Greys')
# #     plt.show()
    




# # %%
# # GAN training
 

# # generator = NN(verbose = False, layerSizes = [50,784], batchSize = 40, learningRate = 1, activation = 'sigmoid', activation_prime = 'sigmoid_prime')
# # discriminator = NN(verbose = False, layerSizes = [784,1], batchSize = 20, learningRate = 1, activation = 'sigmoid', activation_prime = 'sigmoid_prime')
# # generator = NN(verbose = False, layerSizes = [50,100,200,300,784], batchSize = 400, learningRate = 1)
# # discriminator = NN(verbose = False, layerSizes = [784,200,100,50,12], batchSize = 200, learningRate = 1)

# # test plot
# # H = trainingImages[0]
# # plt.imshow(H, interpolation='none', cmap='Greys')
# # plt.show()

# # purposes:
# #   training
# #   test
# #   generated
# # types:
# #   images
# #   digits
# #   inputs
# # %%


# # generate training batch at ratio of GtR (generated to real)
# GtR = 1
# ii = 0
# generatedSize = int(len(trainingImages)*GtR)
# print()
# print('training')
# while True:
#     ii = ii + 1
#     print()
    
#     random.shuffle(trainingData)
#     generatedData = []

#     i = 0
#     for i in range(max(generatedSize-1,100)):
#         # print(i)
#         seed = np.random.rand(40, 1)*0
#         generatedDigit = int(random.random()*10)
#         generatedOutput = np.zeros((10,1))
#         generatedOutput[generatedDigit] = 1.0
#         # print(inputDigit.shape, seed.shape)
#         generatorInput = np.append(generatedOutput, seed)
#         generatedInput = generator.push(generatorInput)

#         generatedData = generatedData + [{'image':copy.deepcopy(generatedInput.reshape(trainingImages[0].shape)),
#             'input':generatedInput,
#             'digit':generatedDigit,
#             'output':generatedOutput,
#             'isGntd': True}]
#         print("\r", 'gi', i, '/', generatedSize, end="")
#         # if i%1000 == 0:
#         #     print('gi: ', i, '/', generatedSize)
#     #print(generatedImages.shape, trainingImages.shape)
#     print()
#     TBDiscriminatedData = generatedData + trainingData[:len(generatedData)]
#     random.shuffle(TBDiscriminatedData)
#     print(ii%50, '/50')
#     # if ii%50 == 0:
#     H = generatedData[0]['image']

#     plt.imshow(H, interpolation='none', cmap='Greys')
#     plt.show()


#     # train discriminator

#     discScore = 0
    
#     i=0
#     avOut = 0

#     for i in range(len(TBDiscriminatedData)):
#         Y = discriminator.push(TBDiscriminatedData[i]['input'])
        
#         # outputDigit = np.zeros((10,1))
#         # outputDigit[TBDiscriminatedData[i]['digit']] = 1.0
        
#         Y_ideal = np.array([float(TBDiscriminatedData[i]['isGntd'])])
        
#         discScore = discScore*0.999 + int((Y[0][0]>0.5)==(TBDiscriminatedData[i]['isGntd']))*0.001
#         avOut = avOut*0.999 + (Y[0][0])*0.001
#         discriminator.acceptFeedback(Y_ideal)

#         print("\r", 'td', i, '/', len(TBDiscriminatedData),'  discScore, avOut: ', round(discScore, 3), avOut, end="")
#         # if i%1000 == 0:
#         #     # print('Y: ', Y)
#         #     # print('Y_ideal: ', Y_ideal)
#         #     # print('td: ',i,'/',len(TBDiscriminatedData),'  discScore: ', discScore)
#         if discScore >= 0.999 :
#             break
#     print()
    
#     if i<len(TBDiscriminatedData)-5:
#         generatedSize = int(generatedSize*0.66)
#     else:
#         generatedSize = min(int(generatedSize*1.5),int(GtR*len(trainingData)))

#     discScore = 1
#     # train generator
#     i = 0
#     nablaAv = discriminator.acceptFeedback(np.array([1]), applyChanges=False)*0
#     # print('nabla: ', nabla)
#     print()
#     avOut = 0
#     while discScore>0.001 and i<50000:
#         i = i + 1
#         seed = np.random.rand(40, 1)*0
#         generatedDigit = int(random.random()*10)
#         inputDigit = np.zeros((10,1))
#         inputDigit[generatedDigit] = 1.0
#         # print(inputDigit.shape, seed.shape)
#         generatorInput = np.append(inputDigit, seed)
#         discriminatorInput = generator.push(generatorInput)
#         Y = discriminator.push(discriminatorInput)
#         discScore = discScore*0.999 + int((Y[0][0]>0.5)==True)*0.001
        
#         nabla = discriminator.acceptFeedback(np.array([1]), applyChanges=False)
#         generator.acceptNabla(nabla, applyChanges=True)
        
#         nablaAv = nablaAv + nabla
#         # if i %100==0:
#         #     print('tg: ', i,'  discScore: ', round(discScore, 3))
        
#         print("\r", 'tg', i, '/', 50000,'  discScore: ', round(discScore, 3), end="")
            
#     # print()
#     print('nablaAv: ')
#     plt.imshow(nablaAv.reshape(trainingImages[0].shape), interpolation='none', cmap='Greys')
#     plt.show()
#     # print()


# # %%    GNN definitions



# # %%    Examine GNN


# testGNN = GNN(inputSize = 2, outputSize = 3, hiddenSize = 3, learningRate = 1, activation = 'swish', activation_prime = 'swish_prime')
# print(testGNN.W)
# print()
# print(testGNN.A[-1])
# print()
# testGNN.tick()
# print(testGNN.W)
# print()
# print(testGNN.A[-1])
# print(relu(np.dot(testGNN.W, testGNN.A[-1])))

# for i in range(1000):
#     testGNN.tick()
#     print(i, testGNN.A[-1].transpose())


# # %%    test that GNN can converge to sigmoid values from linear inputs
# import random
# ticks = 3
# testGNN = GNN(inputSize = 2, outputSize = 3, hiddenSize = 10, learningRate = 0.001, numTrainingIterations = ticks, batchSize = 1, activation = 'swish', activation_prime = 'swish_prime')
# rollingAverage = 1
# limit = 1000
# for i in range(limit):
#     testGNN.learningRate = max((0.3*(limit-i) + i*0.01)/limit, 0.05)
#     rand1 = random.random()
#     rand2 = random.random()
    
#     Y_ideal = np.array([sigmoid(sigmoid(rand1)), 1, sigmoid(1-sigmoid(rand2))])
#     testGNN.push(np.array([rand1, rand2]))
#     for ii in range(ticks-1):
#         out = testGNN.tick().reshape(Y_ideal.shape)
    
#     testGNN.acceptFeedback(Y_ideal)
    
#     score = abs(Y_ideal-out).sum()
#     #print(out, testGNN.W)
#     rollingAverage = (rollingAverage*0.9) + (score*0.1)
    
#     print('score, lr, ra: ', score, testGNN.learningRate, rollingAverage)

#     if rollingAverage<0.05:
#         break

# print(testGNN)
# print('i:',i)

# # %%    perceptron test - make sure it can converge
# perceptron = GNN(inputSize = 1, outputSize = 1, hiddenSize = 0, learningRate = 0.3, batchSize  = 10)
# rollingAverage = 1
# limit = 10000

# B = (1 - 2*random.random())*0.2
# W = (1 - 2*random.random())*0.2

# print('B:', B)
# print('W:', W)

# for i in range(limit):
#     perceptron.learningRate = max((0.03*(limit-i) + i*0.01)/limit, 0.01)
#     X = 1 - 2*random.random()
        
#     Y_ideal = sigmoid(X*W + B)
#     out = perceptron.push(np.array([X]))
    
#     perceptron.acceptFeedback(Y_ideal)
    
#     score = abs(Y_ideal-out).sum()**2

#     rollingAverage = (rollingAverage*0.9) + (score*0.1)
    
#     print('X, out, Y, score, lr, ra: ',X, out,Y_ideal, score, perceptron.learningRate, rollingAverage)

        
#     if rollingAverage<0.025:
#         break
    
# print(perceptron)
# print('i:',i)

# # %%    create a linear transformer

# # inputs:
# #   A0
# # outputs:
# #   Y
# rollingAverage = 1
# ticks = 3
# slope = random.random()
# # add bias here
# retainer = GNN(inputSize = 1, outputSize = 1, hiddenSize = 20, learningRate = 0.05, numTrainingIterations = ticks, batchSize = 1, activation = 'swish', activation_prime = 'swish_prime')
# limit = 1000
# for i in range(limit):
#     retainer.learningRate = max((0.3*(limit-i) + i*0.001)/limit, 0.005)
#     X = (random.random())
        
#     Y_ideal = np.array(X)*slope
#     out = retainer.push(np.array([X]))
#     for ii in range(ticks-1):
#         out = retainer.tick()
    
#     retainer.acceptFeedback(Y_ideal)

#     score = abs(Y_ideal-out).sum()

#     rollingAverage = (rollingAverage*0.9) + (score*0.1)
    
#     print('X, out, Y, score, lr, ra: ', X, out, Y_ideal, score, retainer.learningRate, rollingAverage)

        
#     if rollingAverage<0.005:
#         break
# print(i)

# # %%    create an adder

# # inputs:
# #   X1
# #   X2
# # outputs:
# #   Y
# import copy
# rollingAverage = 1
# prevBest = 1
# ticks = 3
# slope = random.random()
# # add bias here
# adder = GNN(inputSize = 2, outputSize = 1, hiddenSize = 10, learningRate = 0.05, numTrainingIterations = ticks, batchSize = 20, activation = 'swish', activation_prime = 'swish_prime')
# bestNet = copy.deepcopy(adder)
# limit = 2000
# for i in range(limit):
#     adder.learningRate = max((0.3*(limit-i) + i*0.01)/limit, 0.05)
#     X1 = ((random.random()))
#     X2 = ((random.random()))
        
#     Y_ideal = np.array(X1 + X2)
#     out = adder.push(np.array([X1, X2]))
#     for ii in range(ticks-1):
#         out = adder.tick()
    
#     adder.acceptFeedback(Y_ideal)

#     score = abs(Y_ideal-out).sum()

#     rollingAverage = (rollingAverage*0.9) + (score*0.1)
    
#     print('X1, X2, out, Y, score, lr, ra: ', X1, X2, out, Y_ideal, score, adder.learningRate, rollingAverage)

#     if rollingAverage < prevBest:
#         bestNet = copy.deepcopy(adder)
#         prevBest = copy.deepcopy(rollingAverage)

#     # elif rollingAverage > prevBest*1.5:
#     #     adder = copy.deepcopy(bestNet)
#     #     rollingAverage = copy.deepcopy(prevBest)
        
#     if rollingAverage<0.005 or rollingAverage>500:
#         break
# print(i, prevBest)

# print(adder)

# # %%    merge modules, lining up outputs of one with inputs of the next

# # merge([{a: slice(y0, yn), b: slice(x0, xn), c: slice(x0, xn)},{...}]) - pass output of a to inputs of b and c

# # totalSize of MN will be total size of all activation matrices -1perBias -1perOverlappingInput
#     # totalSize = len(a.A[-1]) + len(b.A[-1]) - len(b.A[-1][b.inputSlice]) + len(b.A[-1]) - len(c.A[-1][c.inputSlice])

# def slicelen(s):
#     return s.stop - s.start

# def merge(mergeList):
#     netList = []
#     for d in mergeList:
#         for net in d.keys():
#             # combine biases into 1 array
#             # seperate output and hidden layers from inputs
            
#             pass
    



# # %%    perceptron test - make sure it can converge
# perceptron = GNN(inputSize = 1, outputSize = 1, hiddenSize = 0, learningRate = 0.05, numTrainingIterations = 1, batchSize  = 100, activation = 'swish', activation_prime = 'swish_prime')
# rollingAverage = 1
# limit = 10000


# for i in range(limit):
#     perceptron.learningRate = max((0.1*(limit-i) + i*0.01)/limit, 0.05)
    
#     X = float(random.random()>0.5)*(3.43726-0.125266) + 0.125266
    

#     Y_ideal = X
#     out = perceptron.push(np.array([X]))
    
#     perceptron.acceptFeedback(Y_ideal)
    
#     score = abs(X-out).sum()**2

#     rollingAverage = (rollingAverage*0.9) + (score*0.1)
    
#     print('X, out, Y, score, lr, ra: ',X, out, Y_ideal, score, perceptron.learningRate, rollingAverage)

        
#     if rollingAverage==0:
#         break
    
# #print(perceptron)
# print('i:',i)

# # %% create a 1:1 perceptron
# binary = GNN(inputSize = 1, outputSize = 1, hiddenSize = 0, learningRate = 0.05, numTrainingIterations = 1, batchSize  = 100, activation = 'swish', activation_prime = 'swish_prime')
# binary.W[1,0] = 1.0      # weight
# binary.W[1,2] = 0.101     # bias
# # print(perceptron)
# # print(perceptron.W[1,1])

# resolution = 0.1
# steps = 3
# start = 0
# x1 = 0.12684494889427658
# x2 = 3.4218021060206922

# print(3.43726, float(swish(3.43726)))
# print(0.125266, float(swish(0.125266)))

# breaker = True
# lastx = 0
# i=0
# while breaker:
#     i = i+1
#     X = (np.array(list(range(steps)))*resolution)+start
#     Y = np.array([binary.push(x) for x in X])

#     print('X: ', X)
#     print('Y: ', Y.reshape(X.shape))
#     print()

#     for x, y in zip(X,Y):
        
#         if x == y:
#             lastx = x
#             breaker = False 
#             break

#     for ii in range(steps-1):
        
#         if ((X[ii]>Y[ii]) and (X[ii+1]<Y[ii+1])) or ((X[ii]<Y[ii]) and (X[ii+1]>Y[ii+1])):
#             start = X[ii]
#             resolution = resolution*0.5
# print('*****************************')
# print('i: ', i)
# print('x: ', lastx)
# print('y: ', float(binary.push(x)))
# print(x==y)


# print(binary)

# print('x1, y1:',x1, float(binary.push(x1)))
# print('x2, y2:',x2, float(binary.push(x2)))
"""
# %%
