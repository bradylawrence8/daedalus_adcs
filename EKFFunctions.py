import numpy as np
from adafruit_bno08x.i2c import BNO08X_I2C

#KALMAN FILTER FUNCTIONS
def State7TransitionFcn(x,wm,dt):
    w1,w2,w3 = wm.flatten()-x.flatten()[4:7]
    w_norm = np.sqrt(w1**2+w2**2+w3**2)

    Omega = np.array((0,w3,-w2,w1,
                    -w3,0,w1,w2,
                    w2,-w1,0,w3,
                    -w1,-w2,-w3,0)).reshape(4,4)

    if w_norm*dt/2 < 0.1:
        PHI = np.eye(4) + 0.5*dt*Omega
    else:
    #compute trig terms
        st = np.sin(w_norm*dt/2)
        ct = np.cos(w_norm*dt/2)

        PHI = np.eye(4)*ct+Omega*st/w_norm
        # print("st/w:", st/w_norm)

    qnew = PHI@x[0:4]
    qnew = qnew/np.linalg.norm(qnew)

    bnew = x[4:7]

    xnew = np.zeros((7,1))
    xnew[0:4] = qnew.reshape(4,1)
    xnew[4:7] = bnew.reshape(3,1)

    return xnew

def Measurement7Model(x, sun_Inertial, g=1):
    q1,q2,q3,q4,b1,b2,b3 = x.flatten()
    q_vec = x[0:3]

    Omega = np.array([[0,q3,-q2],[-q3,0,q1],[q2,-q1,0]])

    DCM = (q4**2-q_vec.T@q_vec)*np.eye(3)+2*q_vec@q_vec.T+2*q4*Omega

    # print("DCM:", DCM)

    g_dir = np.array([[0],[0],[g]])
    a_body = DCM.T@g_dir

    s_body = DCM.T@sun_Inertial
    s_body /= np.linalg.norm(s_body)

    h = np.vstack((a_body,s_body))

    return h

def F7_Jacobian(x,w,dt):
    q1,q2,q3,q4,b1,b2,b3 = x.flatten()
    w1,w2,w3 = w.flatten()-x.flatten()[4:7]
    w_norm = np.sqrt(w1**2 + w2**2 + w3**2)

    Omega = np.array((0,w3,-w2,w1,
                    -w3,0,w1,w2,
                    w2,-w1,0,w3,
                    -w1,-w2,-w3,0)).reshape(4,4)

    if w_norm < 1e-6:
        PHI = np.eye(4) + 0.5*dt*Omega
    else:
    #compute trig terms
        st = np.sin(w_norm*dt/2)
        ct = np.cos(w_norm*dt/2)

        PHI = np.eye(4)*ct+Omega*st/w_norm

    qbias = 0.5*dt*np.array((q4,-q3,q2,
                             q3,q4,-q1,
                             -q2,q1,q4,
                             -q1,-q2,-q3)).reshape(4,3)

    F = np.zeros((7,7))
    F[0:4,0:4] = PHI
    F[0:4,4:7] = qbias
    F[4:7,4:7] = np.eye(3)
    return F

def H7_Jacobian(x, s_vec, g=9.81):
    q1,q2,q3,q4,b1,b2,b3 = x.flatten()
    s1,s2,s3 = s_vec.flatten()

    H = np.zeros((6,7))

    H[0,0] = 2*g*q3
    H[0,1] = 2*g*q4
    H[0,2] = 2*g*q1
    H[0,3] = 2*g*q2

    H[1,0] = -2*g*q4
    H[1,1] = 2*g*q3
    H[1,2] = 2*g*q2
    H[1,3] = -2*g*q1

    H[2,0] = -2*g*q1
    H[2,1] = -2*g*q2
    H[2,2] = 2*g*q3
    H[2,3] = 2*g*q4

    t1 = 2*q1*s1+2*q2*s2+2*q3*s3
    t2 = 2*q2*s3-2*q3*s2+2*q4*s1
    t3 = 2*q3*s1-2*q1*s3+2*q4*s2
    t4 = 2*q1*s2-2*q2*s1+2*q4*s3

    H[3,0] = t1
    H[3,1] = t4
    H[3,2] = 2*q1*s3-2*q3*s1-2*q4*s2
    H[3,3] = t2

    H[4,0] = 2*q2*s1-2*q1*s2-2*q4*s3
    H[4,1] = t1
    H[4,2] = t2
    H[4,3] = t3

    H[5,0] = t3
    H[5,1] = 2*q3*s2-2*q2*s3-2*q4*s1
    H[5,2] = t1
    H[5,3] = t4

    return H



#Attitude Conversions
def YPR_to_q(YPR):
    Y, P, R = YPR.flatten()

    #precompute sin and cos
    sR = np.sin(R/2)
    cR = np.cos(R/2)
    sP = np.sin(P/2)
    cP = np.cos(P/2)
    sY = np.sin(Y/2)
    cY = np.cos(Y/2)

    q = np.array((sR*cP*cY-cR*sP*sY,
                  sR*cP*sY+cR*sP*cY,
                  cR*cP*sY-sR*sP*cY,
                  cR*cP*cY+sR*sP*sY)).reshape(4,1)

    return q

def q_to_YPR(q):
    q1, q2, q3, q4 = q.flatten()
    q_vec = np.array((q1,q2,q3)).reshape(3,1)

    Omega = np.array([[0,q3,-q2],[-q3,0,q1],[q2,-q1,0]])

    DCM = (q4**2-q_vec.T@q_vec)*np.eye(3)+2*q_vec@q_vec.T+2*q4*Omega

    #Check This
    pitch = np.arcsin(-DCM[0,2])
    roll  = np.arctan2(DCM[0,1], DCM[0,0])
    yaw   = np.arctan2(DCM[1,2], DCM[2,2])

    YPR = np.array((yaw,pitch,roll)).reshape(3,1)

    return YPR

def qMultiply(q,q_p):
    q_p1, q_p2, q_p3, q_p4 = q_p.flatten()

    DCM = np.array((q_p4, q_p3, -q_p2, q_p1,
                    -q_p3, q_p4, q_p1, q_p2,
                    q_p2, -q_p1, q_p4, q_p3,
                    -q_p1, -q_p2, -q_p3, q_p4)).reshape(4,4)
    q_pp = DCM@q
    return q_pp

def q_inv(q):
    q_vec = np.array([[q[0]],[q[1]],[q[2]]])
    return np.append(-q_vec,q[3])


def R1(x):
    R1 = np.array([[1,0,0],[0,np.cos(x),np.sin(x)],[0,-np.sin(x),np.cos(x)]])
    return R1

def R2(x):
    R2 = np.array([[np.cos(x),0,-np.sin(x)],[0,1,0],[np.sin(x),0,np.cos(x)]])
    return R2

def R3(x):
    R3 = np.array([[np.cos(x),np.sin(x),0],[-np.sin(x),np.cos(x),0],[0,0,1]])
    return R3

def DCM(w,dt):
    wx, wy, wz = w.flatten()
    DCM = R3(wz*dt)@R2(wy*dt)@R1(wx*dt)
    return DCM


#READ IMU
def readAccel(bno):
    ax, ay, az = bno.acceleration
    return np.array([[ax],[ay],[az]])


def readGyro(bno):
    gx, gy, gz = bno.gyro
    return np.array([[gx],[gy],[gz]])
