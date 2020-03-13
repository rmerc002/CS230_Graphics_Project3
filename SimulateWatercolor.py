import numpy as np
import os
import sys
import cv2
class Watercolor:
    def __init__(self,rows=40,cols=40,numPigments=1,T=10):
        self.rows = rows
        self.cols = cols
        self.T = T
        self.numPigments = numPigments

        self.h = np.zeros((rows,cols))
        # self.h /= self.h.max()
        # cv2.imshow("paper",self.h)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.M = np.ones((rows,cols))
        self.u = np.zeros((rows,cols+1))
        self.v = np.zeros((rows+1,cols))
        self.p = np.zeros((rows,cols))
        self.g = np.ones((rows,cols,numPigments))
        self.d = np.zeros((rows,cols,numPigments))
        self.s = np.zeros((rows,cols))

        self.mu = 0.1
        # self.kappa = 0.01
        self.kappa = 0.1

        self.im = np.zeros((rows,cols,3))

        # self.pigmentParameters = np.zeros((numPigments,9))
        self.pigmentParameters = np.array([ [0.22, 1.47, 0.57, 0.050, 0.003, 0.030, 0.02, 5.5, 0.81],
                                            [0.46, 1.07, 1.50, 1.280, 0.380, 0.210, 0.05, 7.0, 0.40],
                                            [0.10, 0.36, 3.45, 0.970, 0.650, 0.007, 0.05, 3.4, 0.81],
                                            [1.62, 0.61, 1.64, 0.010, 0.012, 0.003, 0.09, 1.0, 0.41],
                                            [1.52, 0.32, 0.25, 0.060, 0.260, 0.400, 0.01, 1.0, 0.31],
                                            [0.74, 1.54, 2.10, 0.090, 0.090, 0.004, 0.09, 9.3, 0.90],
                                            [0.14, 1.08, 1.68, 0.770, 0.015, 0.018, 0.02, 1.0, 0.63],
                                            [0.13, 0.81, 3.45, 0.005, 0.009, 0.007, 0.01, 1.0, 0.14],
                                            [0.06, 0.21, 1.78, 0.500, 0.880, 0.009, 0.06, 1.0, 0.08],
                                            [1.55, 0.47, 0.63, 0.010, 0.050, 0.035, 0.02, 1.0, 0.12],
                                            [0.86, 0.86, 0.06, 0.005, 0.005, 0.090, 0.01, 3.1, 0.91],
                                            [0.08, 0.11, 0.07, 1.250, 0.420, 1.430, 0.06, 1.0, 0.08]])


    def MainLoop(self):
        t=0
        # with np.printoptions(precision=3, suppress=True):
        #     print("temp u at t={}, sum={}: \n{}".format(t,self.u.sum(),self.u))
        #     # print("temp v at t={}: \n{}".format(t,self.v))
        #     print("###############################")
        for t in range(self.T):
            print("Iterations with t={}/{}".format(t,self.T))
            if t > 0:
                # print("before move water u: {}".format(self.u[0,:]))
                self.MoveWater()
                # print("move water u: {}".format(self.u[0,:]))
                self.MovePigment()
                # print("move pigment u: {}".format(self.u[0,:]))
                self.TransferPigment()
                # print("transfer pigment u: {}".format(self.u[0,:]))
                self.SimulateCapillaryFlow()
                # print("simulatecapillaryflow u: {}".format(self.u[0,:]))
            # with np.printoptions(precision=3, suppress=True):
            #     print("temp u at t={}, sum={}: \n{}".format(t,self.u.sum(),self.u))
            #     # print("temp v at t={}: \n{}".format(t,self.v))
            #     print("###############################")

            # ushow = self.u
            # ushow = self.u[:,:-1] * self.v[:-1,:]
            # ushow = np.minimum(ushow,np.ones(ushow.shape))
            # ushow = np.maximum(ushow,-1*np.ones(ushow.shape))
            # ushow = (ushow+1)/2
            # cv2.imshow("u velocity",cv2.resize(ushow,(500,500)))
            # cv2.waitKey(300)

            # with np.printoptions(precision=12, suppress=True):
            #     print("Image rows:")
            #     for r in range(self.rows):
            #         print(self.u[r,:])
                # cv2.destroyAllWindows()

            self.im = self.RenderLayer()
            # with np.printoptions(precision=1, suppress=True):
            #     print("Image rows:")
            #     for r in range(self.rows):
            #         print("t={},r={},\n{}".format(t,r,self.im[r,:]))
            cv2.imshow("rendered image",cv2.resize(self.im,(500,500)))
            # cv2.imshow("u velocity",self.im)

            cv2.waitKey(300)


        # with np.printoptions(precision=3, suppress=True):
        #     print("final u: \n{}".format(self.u))
        #     # print("final v: \n{}".format(self.v))
        # cv2.destroyAllWindows()
        print("finished iterating with T={}".format(self.T))
        self.im = self.RenderLayer()
        return self.im

    def MoveWater(self):
        self.UpdateVelocities()
        self.RelaxDivergence()
        self.FlowOutward()

    def UpdateVelocities(self):

        self.u[:,1:-2] -= (self.h[:,2:] - self.h[:,0:-2])/2#TODO
        self.u[:,0] -= (self.h[:,1] - self.h[:,0])
        self.u[:,-1] -= (self.h[:,-1] - self.h[:,-2])

        self.v[1:-2,:] -= (self.h[2:,:] - self.h[:-2,:])/2 #TODO
        self.v[0,:] -= (self.h[1,:] - self.h[0,:])
        self.v[0,:] -= (self.h[-2,:] - self.h[-1,:])
        delT = self.getDelT()
        # print("delT: {}".format(delT))



        u = np.zeros(self.u.shape)
        v = np.zeros(self.v.shape)
        for t in np.linspace(0,1,int(np.floor(1/delT))):
            for r in range(self.rows):
                # print("in deltaT loop r={}, u: {}".format(r, self.u[0,:]))
                for c in range(self.cols):
                    if c > 0:
                        temp1 = np.mean([self.u[r,c-1],self.u[r,c]])
                        temp1 = temp1 * temp1
                    else:
                        temp1 = 0
                    if c < self.cols-1:
                        temp2 = np.mean([self.u[r,c],self.u[r,c+1]])
                        temp2 = temp2 * temp2
                    else:
                        temp2 = 0
                    if (r > 0 and c < self.cols-1):
                        temp3 = np.mean([self.u[r-1,c],self.u[r,c]]) * np.mean([self.v[r-1,c],self.v[r-1,c+1]])
                    else:
                        temp3 = 0
                    if r < self.rows-1 and c < self.cols-1:
                        temp4 = np.mean([self.u[r,c],self.u[r+1,c]]) * np.mean([self.v[r,c],self.v[r,c+1]])
                    else:
                        temp4 = 0
                    A = temp1 - temp2 + temp3 - temp4

                    if c < self.cols-1:
                        temp1 = self.u[r,c+1]
                    else:
                        temp1 = 0
                    if c > 0:
                        temp2 = self.u[r,c-1]
                    else:
                        temp2 = 0
                    if r < self.rows-1:
                        temp3 = self.u[r+1,c]
                    else:
                        temp3 = 0
                    if r > 0:
                        temp4 = self.u[r-1,c]
                    else:
                        temp4 = 0
                    temp5 = self.u[r,c]
                    B = temp1 + temp2 + temp3 + temp4 - 4*temp5
                    # B = 0
                    A = 0
                    tempPressure = self.p[r,c]
                    if c < self.cols-1:
                        tempPressure -= self.p[r,c+1]
                    u[r,c] = temp5 + delT*(A + self.mu*B + tempPressure - self.kappa*temp5)
                    # u[r,c] = temp5 + delT*A
                    # u[r,c] = temp5 + delT*self.mu*B


                    if r > 0:
                        temp1 = np.mean([self.v[r-1,c],self.v[r,c]])
                        temp1 = temp1 * temp1
                    else:
                        temp1 = 0
                    if r < self.rows-1:
                        temp2 = np.mean([self.v[r,c],self.v[r+1,c]])
                        temp2 = temp2 * temp2
                    else:
                        temp2 = 0
                    if (c > 0 and r < self.rows-1):
                        temp3 = np.mean([self.v[r,c-1],self.v[r,c]]) * np.mean([self.u[r,c-1],self.u[r+1,c-1]])
                    else:
                        temp3 = 0
                    if c < self.cols-1 and r < self.rows-1:
                        temp4 = np.mean([self.v[r,c],self.v[r,c+1]]) * np.mean([self.u[r,c],self.u[r+1,c]])
                    else:
                        temp4 = 0
                    A = temp1 - temp2 + temp3 - temp4

                    if c < self.cols-1:
                        temp1 = self.v[r,c+1]
                    else:
                        temp1 = 0
                    if c > 0:
                        temp2 = self.v[r,c-1]
                    else:
                        temp2 = 0
                    if r < self.rows-1:
                        temp3 = self.v[r+1,c]
                    else:
                        temp3 = 0
                    if r > 0:
                        temp4 = self.v[r-1,c]
                    else:
                        temp4 = 0
                    temp5 = self.v[r,c]
                    B = temp1 + temp2 + temp3 + temp4 - 4*temp5
                    # B = 0
                    A = 0
                    tempPressure = self.p[r,c]
                    if r < self.rows-1:
                        tempPressure -= self.p[r+1,c]
                    v[r,c] = temp5 + delT*(A + self.mu*B + tempPressure - self.kappa*temp5)
            self.u = u
            self.v = v
            self.EnforceBoundaryConditions()
        return

    def getDelT(self):
        # print("u: {}".format(self.u))
        # print("v: {}".format(self.v))
        # print("max u: {}, max v: {}".format(np.abs(self.u).max(), np.abs(self.v).max()))
        # print("getDelT: {}".format(np.ceil(np.maximum(np.abs(self.u).max(),np.abs(self.v).max()) )) )
        return 1/np.ceil(np.maximum(np.abs(self.u).max(),np.abs(self.v).max()))


    def EnforceBoundaryConditions(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.M[row,col] == 0:
                    if col > 0 and self.M[row,col-1] == 0:
                        if col < self.col-1 and self.M[row,col+1] == 0:
                            self.u[row,col] = 0
                    if row > 0 and self.M[row-1,col] == 0:
                        if row < self.row-1 and self.M[row+1,col] == 0:
                            self.v[row,col] = 0

    def RelaxDivergence(self):
        N = 50
        tau = 0.01
        epsilon = 0.1# 0.1
        t = 0
        u = self.u.copy()
        v = self.v.copy()
        while True:
            deltaMax = 0
            for row in range(self.rows):
                for col in range(self.cols):
                    delta = -epsilon*(self.u[row,col+1] - self.u[row,col] + self.v[row+1,col] - self.v[row,col])
                    # m = (self.u[row,col+1] + self.u[row,col] + self.v[row+1,col] + self.v[row,col])/4
                    # self.p[row,col] += delta
                    u[row,col+1] += delta
                    u[row,col] -= delta
                    v[row+1,col] += delta
                    v[row,col] -= delta

                    u[row,col+1]
                    deltaMax = max(abs(delta), deltaMax)
            # print("t={},deltaMax: {}".format(t,deltaMax))
            self.u = u.copy()
            self.v = v.copy()
            if t >= N or deltaMax <= tau:
                # print("\n\t\t!!!!#!#***BREAK***#!#!!!\n")
                break
            t += 1




    def FlowOutward(self):

        pass


    def MovePigment(self):
        delT = self.getDelT()
        # print("delT: {}".format(delT))
        for k in range(self.numPigments):
            for t in np.linspace(0,1,int(np.floor(1/delT))):
                g = self.g[:,:,k]
                gp = g
                for r in range(self.rows):
                    for c in range(self.cols):
                        totalOut = 0
                        if c < self.cols-1:
                            totalOut += max(0, self.u[r,c+1]*g[r,c])
                        if c > 0:
                            totalOut += max(0, -self.u[r,c]*g[r,c])
                        if r < self.rows-1:
                            totalOut += max(0, self.v[r+1,c]*g[r,c])
                        if r > 0:
                            totalOut += max(0, -self.v[r,c]*g[r,c])

                        # totalOut = 0.0001
                        scaleCorrection = 0.99999
                        if totalOut > 0:
                            scaleCorrection = 0.99999*min(1, gp[r,c]/totalOut)

                        if c < self.cols-1:
                            gp[r,c+1] += max(0,self.u[r,c+1]*g[r,c])*scaleCorrection
                        if c > 0:
                            gp[r,c-1] += max(0,-self.u[r,c]*g[r,c])*scaleCorrection
                        if r < self.rows-1:
                            gp[r+1,c] += max(0,self.v[r+1,c]*g[r,c])*scaleCorrection
                        if r > 0:
                            gp[r-1,c] += max(0,-self.v[r,c]*g[r,c])*scaleCorrection
                        # gp[r,c] = gp[r,c] - min(gp[r,c], max(0, self.u[r,c]*g[r,c]) + max(0,-self.u[r,c-1]*g[r,c]) + max(0,self.v[r,c]*g[r,c]) + max(0,-self.v[r-1,c]*g[r,c]))
                        gp[r,c] = gp[r,c] - totalOut*scaleCorrection

                        if gp[r,c] < 0:
                            print("gp: {}".format(gp[r,c]))
                self.g[:,:,k] = gp


    def TransferPigment(self):
        for k in range(self.numPigments):
            (ro,omega,gamma) = self.pigmentParameters[k,-3:]

            for r in range(self.rows):
                for c in range(self.cols):
                    if self.M[r,c] == 1:
                        deltaDown = self.g[r,c,k]*(1-self.h[r,c]*gamma)*ro
                        deltaUp = self.d[r,c,k]*(1+(self.h[r,c]-1)*gamma)*ro/omega
                        # print("deltaDown: {}, deltaUp: {}, self.d[r,c,k]".format(deltaDown,deltaUp,self.d[r,c,k]))
                        if self.d[r,c,k] + deltaDown > 1:
                            deltaDown = max(0, 1-self.d[r,c,k])
                        if self.g[r,c,k] + deltaUp > 1:
                            deltaUp = max(0, 1-self.g[r,c,k])
                        self.d[r,c,k] += deltaDown - deltaUp
                        self.g[r,c,k] += deltaUp - deltaDown
                        with np.printoptions(precision=3, suppress=True):
                            if self.d[r,c,k] < 0 or self.g[r,c,k] < 0:
                                print("d: {:.3}, g: {:.3}, dUP: {:.3}, dDOWN: {:.3}".format(self.d[r,c,k], self.g[r,c,k], deltaUp, deltaDown))

    def SimulateCapillaryFlow(self):

        pass

    def ReadImageData(self,inputPath):


        imageTypes = [  "pigment",
                        "heightfield",
                        "wetareamask",
                        "xvelocity",
                        "yvelocity",
                        "waterpressure",
                        "pigmentconcentration",
                        "pigmentdeposited",
                        "watersaturation",
                        ]
        fileNames={type:None for type in imageTypes}
        for root, dirs, files in os.walk(inputPath):
            for name in files:
                for imageType in imageTypes:
                    if imageType in name.lower():
                        fileNames[imageType] = os.path.join(root,name)



        numPigments = self.pigmentParameters.shape[1]

        if fileNames["heightfield"] is not None:
            # with open(fileNames["heightfield"][0]) as file:
            self.h = np.load(fileNames["heightfield"])

        (self.rows,self.cols) = self.h.shape
        rows = self.rows
        cols = self.cols



        if fileNames["wetareamask"] is not None:
            self.M = np.load(fileNames["wetareamask"])
        ### u,v: velocities in the x,y directions
        if fileNames["xvelocity"] is not None:
            self.u = np.load(fileNames["xvelocity"])
            # print(self.u[0,:])

        if fileNames["yvelocity"] is not None:
            self.v = np.load(fileNames["yvelocity"])
            # print(self.v[0,:])
        ### p: pressure of the water
        # if fileNames["waterpressure"] is not None:
        self.p = np.zeros((rows,cols))
        ### g: concentations of each pigment
        if fileNames["pigmentconcentration"] is not None:
            self.g = np.load(fileNames["pigmentconcentration"])
        ### d: deposited pigment
        # if fileNames["pigmentdeposited"] is not None:
        self.d = np.zeros((rows,cols,numPigments))
            # print("pigmentdeposited = {}".format(fileNames["pigmentdeposited"])))


        ### s: water saturation
        # if fileNames["watersaturation"] is not None:
        self.s = np.zeros((rows,cols))



        return

    def RenderLayer(self):
        #details in section 5
        #Kubelka-Munk color model

        (rows,cols,numPigments) = self.d.shape
        x = self.g + self.d

        # print("x size: {}".format(x.shape))
        R1 = np.ones((rows,cols,3))
        T1 = np.zeros((rows,cols,3))
        Rw = np.array([0.3,0.3,0.7])
        Rb = np.array([0.25,0.25,0.6])
        # a = 0.5*(Rw+(Rb-Rw+1)/Rb)
        pigmentIndex = 0
        K = self.pigmentParameters[pigmentIndex,0:3]
        S = self.pigmentParameters[pigmentIndex,3:6]
        a = (S+K)/S
        # a = K/S + 1
        b = np.sqrt(a*a-1)
        # trigTerm = (b*b-(a-Rw)*(a-1))/(b*(1-Rw))
        # S = (1/b)*np.arccosh(trigTerm)/np.arcsinh(trigTerm)
        # K = S*(a-1)

        R2 = 0.9*np.ones((rows,cols,3))#canvas

        T2 = np.zeros((rows,cols,3))#canvas
        p = 0
        # for row in range(rows):
        #     for col in range(cols):
        #         innerTerm = b*S*x[row,col,p]
        #         if(np.any(np.abs(innerTerm) > 10)):
        #             R1[row,col,:] = 0
        #             T1[row,col,:] = 1
        #         else:
        #             aTerm = a*np.sinh(innerTerm)
        #             bTerm = b*np.cosh(innerTerm)
        #             c = aTerm + bTerm
        #
        #             R1[row,col,:] = np.sinh(innerTerm/c)
        #             if np.any(R1[row,col,:] > 1):
        #                 print("\t\t DANGER !!!")
        #     # print("row: {},R: {}".format(row,R[row,0,:]))
        #             T1[row,col,:] = b/c
        # print("inner Term: {}".format(b*S*100))
        for row in range(rows):
            for col in range(cols):
                innerTerm = b*S*x[row,col,p]
                if False and (np.any(np.abs(innerTerm) > 10)):
                    R1[row,col,:] = 0
                    T1[row,col,:] = 0
                else:
                    aTerm = a*np.sinh(innerTerm)
                    bTerm = b*np.cosh(innerTerm)
                    c = aTerm + bTerm

                    R1[row,col,:] = np.sinh(innerTerm)/c
                    # print("r= {}, c= {}, aTerm = {}, bTerm = {}, c= {}, innerTerm = {}".format(row,col,np.mean(aTerm),np.mean(bTerm),np.mean(c),np.mean(innerTerm)))
                    if np.any(R1[row,col,:] > 1):
                        print("\t\t DANGER !!!")
            # print("row: {},R: {}".format(row,R[row,0,:]))
                    T1[row,col,:] = b/c


        R2 = R1 + (T1*T1*R2)/(1-R1*R2)
        # R2 = np.minimum(np.ones((rows,cols,3)),R2)
        # R2 = np.maximum(np.zeros((rows,cols,3)),R2)
        T2 = (T1*T2)/(1-R1*R2)
        # show = R2
        # show = np.minimum(ushow,np.ones(ushow.shape))
        # show = np.maximum(ushow,-1*np.ones(ushow.shape))
        # show = (ushow+1)/2
        # with np.printoptions(precision=3, suppress=True):
        #     print("final R: \n{}".format(show[:,:,0]))

        return R2

    def WriteImageData(self, im, inputPath):
        cv2.imwrite(os.path.join(inputPath,"output.png"),cv2.resize(im,(500,500))*255)

def testStroke(painting):
    rows = painting.rows
    cols = painting.cols
    painting.g = 0*np.ones((rows,cols,1))
    painting.g[int(rows/2),:,0] = 1
    painting.g[:,int(cols/2),0] = 1

    sinAngles = np.zeros((rows,cols))
    cosAngles = np.zeros((rows,cols))
    radiuses = np.zeros((rows,cols))
    rh = int(rows/2)
    ch = int(cols/2)
    for r in range(rows):
        for c in range(cols):
            radiuses[r,c] = max(1,np.sqrt((r-rh)*(r-rh) + (c-ch)*(c-ch)))
            # print("r: {}, rh: {}, c: {}, ch: {}, arcsin({}), arccos({}), radius: {}".format(r, rh, c, ch, (r-rh)/radiuses[r,c],(c-ch)/radiuses[r,c],radiuses[r,c]))
            sinAngles[r,c] = np.arcsin((r-rh)/radiuses[r,c])
            cosAngles[r,c] = np.arccos((c-ch)/radiuses[r,c])
    painting.u = np.ones((rows,cols+1))
    painting.v = np.ones((rows+1,cols))

    painting.u[:,:-1] = np.sin(sinAngles)*radiuses/ch
    painting.v[:-1,:] = -np.cos(cosAngles)*radiuses/rh
    painting.u[np.isnan(painting.u)] = 0
    painting.v[np.isnan(painting.v)] = 0
    #
    # painting.u *= -1
    # painting.v *= -1

    # print("u: {}".format(painting.u))
    # print("v: {}".format(painting.v))
    return painting

if __name__ == "__main__":

    # np.set_printoptions(precision=2)
    #arg for the folder with the input files to render
    #the input files have the wetnes mask and pigment info
    inputPath = sys.argv[1]

    ### M: wet area
    ### u,v: velocities in the x,y directions
    ### p: pressure of the water
    ### g: concentations of each pigment
    ### d: depositied pigment
    ### s: water saturation
    firstPainting = Watercolor()
    firstPainting.ReadImageData(inputPath)

    # firstPainting = testStroke(firstPainting)

    im = firstPainting.MainLoop()
    firstPainting.WriteImageData(im, inputPath)
