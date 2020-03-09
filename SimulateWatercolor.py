import numpy as np
import os
import sys
import cv2
class Watercolor:
    def __init__(self,rows=40,cols=40,numPigments=1,T=100):
        self.rows = rows
        self.cols = cols
        self.T = T

        self.h = np.random.normal(0.5, 0.1,rows*cols).reshape((rows,cols))
        self.h -= self.h.min()
        self.h /= self.h.max()
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
        self.kappa = 0#0.1

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
        ### for testing
        self.u[int(self.u.shape[0]/2),int(self.u.shape[1]/2)] = 1
        self.v[int(self.v.shape[0]/2),int(self.v.shape[1]/2)] = 1

    def MainLoop(self):
        t=0
        with np.printoptions(precision=3, suppress=True):
            print("temp u at t={}, sum={}: \n{}".format(t,self.u.sum(),self.u))
            print("temp v at t={}: \n{}".format(t,self.v))
            print("###############################")
        for t in range(self.T):
            self.MoveWater()
            self.MovePigment()
            self.TransferPigment()
            self.SimulateCapillaryFlow()
            with np.printoptions(precision=3, suppress=True):
                print("temp u at t={}, sum={}: \n{}".format(t,self.u.sum(),self.u))
                print("temp v at t={}: \n{}".format(t,self.v))
                print("###############################")
                ushow = self.u-np.min(self.u)
                ushow = ushow/np.max(ushow)
                cv2.imshow("u velocity",cv2.resize(ushow,(500,500)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        with np.printoptions(precision=3, suppress=True):
            print("final u: \n{}".format(self.u))
            print("final v: \n{}".format(self.v))
        print("finished iterating with T={}".format(self.T))
        self.im = self.RenderLayer()
        return self.im

    def MoveWater(self):
        self.UpdateVelocities()
        self.RelaxDivergence()
        self.FlowOutward()

    def UpdateVelocities(self):
        self.u = self.u #TODO
        self.v = self.v #TODO
        delT = 1/np.ceil(np.maximum(np.abs(self.u).max(),np.abs(self.v).max()))


        u = np.zeros(self.u.shape)
        v = np.zeros(self.v.shape)
        for t in np.linspace(0,1,int(np.floor(1/delT))):
            for r in range(self.rows):
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
                    # A = 0
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
                    # A = 0
                    tempPressure = self.p[r,c]
                    if r < self.rows-1:
                        tempPressure -= self.p[r+1,c]
                    v[r,c] = temp5 + delT*(A + self.mu*B + tempPressure - self.kappa*temp5)
            self.u = u
            self.v = v
            self.EnforceBoundaryConditions()
        return

    def EnforceBoundaryConditions(self):
        pass

    def RelaxDivergence(self):

      pass


    def FlowOutward(self):

        pass


    def MovePigment(self):

        pass


    def TransferPigment(self):

        pass


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
        fileNames={type:[] for type in imageTypes}
        for root, dirs, files in os.walk(inputPath):
            for name in files:
                for imageType in imageTypes:
                    if imageType in name.lower():
                        fileNames[imageType].append(os.path.join(root,name))



        numPigments = self.pigmentParameters.shape[1]

        if len(fileNames["heightfield"]) > 0:
            self.h = cv2.imread(fileNames["heightfield"], flags=cv2.IMREAD_COLOR)

        (rows,cols) = self.h.shape



        if len(fileNames["wetareamask"]) > 0:
            self.M = cv2.imread(fileNames["wetareamask"], flags=cv2.IMREAD_COlOR)
        ### u,v: velocities in the x,y directions
        if len(fileNames["xvelocity"]) > 0:
            self.u = cv2.imread(fileNames["xvelocity"], flags=cv2.IMREAD_COlOR)

        if len(fileNames["yvelocity"]) > 0:
            self.v = cv2.imread(fileNames["yvelocity"], flags=cv2.IMREAD_COlOR)
        ### p: pressure of the water
        if len(fileNames["waterpressure"]) > 0:
            self.p = np.zeros((rows,cols))
        ### g: concentations of each pigment
        if len(fileNames["pigmentconcentration"]) > 0:
            self.g = cv2.imread(fileNames["pigmentconcentration"], flags=cv2.IMREAD_COlOR)
        ### d: deposited pigment
        if len(fileNames["pigmentdeposited"]) > 0:
            self.d = np.zeros((rows,cols,numPigments))
            print("pigmentdeposited = {}".format(len(fileNames["pigmentdeposited"])))

        else:
            self.d = np.zeros((rows,cols,1))
            for row in range(rows):
                self.d[row,:] = row/rows;
            print("initializing d")
        ### s: water saturation
        if len(fileNames["watersaturation"]) > 0:
            self.s = np.zeros((rows,cols))



        return

    def RenderLayer(self):
        #details in section 5
        #Kubelka-Munk color model

        (rows,cols,numPigments) = self.d.shape
        x = self.g + self.d
        print("x size: {}".format(x.shape))
        R = np.zeros((rows,cols,3))
        T = np.zeros((rows,cols,3))
        Rw = np.array([0.1,0.1,0.7])
        Rb = np.array([0.05,0.05,0.3])
        a = 0.5*(Rw+(Rb-Rw+1)/Rb)
        b = np.sqrt(a*a-1)
        trigTerm = (b*b-(a-Rw)*(a-1))/(b*(1-Rw))
        S = (1/b)*np.arcsinh(trigTerm)/np.arccosh(trigTerm)
        K = S*(a-1)

        p = 0
        for row in range(rows):
            for col in range(cols):
                c = a*np.sinh(b*S*x[row,col,p]) + b*np.cosh(b*S*x[row,col,p])
                R[row,col,:] = np.sinh(b*S*x[row,col,p]/c)
            # print("row: {},R: {}".format(row,R[row,0,:]))
            # T = b./c
        return R

    def WriteImageData(self, im, inputPath):
        cv2.imwrite(os.path.join(inputPath,"output.png"),self.im*255)

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
    firstPainting = Watercolor(rows=9,cols=9)
    firstPainting.ReadImageData(inputPath)
    im = firstPainting.MainLoop()
    firstPainting.WriteImageData(im, inputPath)
