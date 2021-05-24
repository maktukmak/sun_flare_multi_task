import numpy as np
from scipy.linalg import norm
from scipy.special import softmax, expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def class_log_nll_single(pars):

    Theta_s = pars[0:m]
    b_s = pars[m]
    
    nll = 0
    mu = expit(x @ Theta_s + b_s)    
    nll += -np.sum(y * np.log(mu + 1e-5) + (1-y)*np.log(1-mu + 1e-5))
    nll += sum((Theta_s)**2) / (2 * s_s)
    return nll

def class_log_nll_single_der(pars):
    
    Theta_s = pars[0:m]
    b_s = pars[m]
    
    mu = expit(x @ Theta_s + b_s)
    grad_th = ((y - mu)[None].T * x).sum(axis = 0) - Theta_s/s_s
    grad_b = (y - mu).sum()
    grad = np.concatenate((grad_th, grad_b[None]))
    return -grad

def class_log_nll_multi(pars):

    Theta_s = pars[0:m]
    b_s = pars[m]
    Theta_k = pars[m+1:].reshape(K,m+1)[:, :m]
    b_k = pars[m+1:].reshape(K,m+1)[:, m]
    
    nll = 0
    nll += sum(Theta_s**2) / (2*s_s)
    for k in range(len(Theta_k)):
        mu = expit(x[k] @ Theta_k[k] + b_k[k])    
        nll += -np.sum(y[k] * np.log(mu + 1e-5) + (1-y[k])*np.log(1-mu + 1e-5))
        nll += sum((Theta_k[k] - Theta_s)**2) / (2 * s_k)
    return nll

def class_log_nll_multi_der(pars):
    
    Theta_s = pars[0:m]
    b_s = pars[m]
    Theta_k = pars[m+1:].reshape(K,m+1)[:, :m]
    b_k = pars[m+1:].reshape(K,m+1)[:, m]
    
    grad_s = (Theta_k - Theta_s).sum(axis = 0) / s_k - Theta_s/s_s
    grad_s = np.concatenate((grad_s, [0]))
    
    grad_k = []
    for k in range(len(Theta_k)):
        mu = expit(x[k] @ Theta_k[k] + b_k[k])
        grad_th = ((y[k] - mu)[None].T * x[k]).sum(axis = 0) - (Theta_k[k] - Theta_s)/s_k
        grad_b = (y[k] - mu).sum()
        grad_k.append(np.concatenate((grad_th, grad_b[None])))
        
    grad = np.concatenate((grad_s, np.array(grad_k).flatten()))
    return -grad


with open("data_for_multitask.txt", "rb") as fp:
    x_m, y_m, x, y, xte, yte = pickle.load(fp) 

common_feats = [3, -3, -2, -1]

x = x[:,common_feats]
xte = xte[:,common_feats]

J = len(x)
m = x.shape[1]
K = 2
s_k = 0.1
s_s = 1

x_m = (x_m - x_m.mean(axis = 0)) / x_m.std(axis = 0)
x_mean = x.mean(axis = 0)
x_std =  x.std(axis = 0)
x = (x - x_mean) / x_std
xte = (xte - x_mean) / x_std


clf = LogisticRegression(random_state=0).fit(x, y)
#clf = SVC(gamma='auto').fit(x, y)
yp = clf.predict(xte)
tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
acc_l = (tp + tn) / (tn + fp + fn + tp)
hss2_l = (2 * ((tp * tn) - (fn*fp))) / ((tp + fn) * (fn + tn) + (tn + fp) * (tp + fp))

acc_s_vec = []
s_kv = [0.01, 0.1, 1, 10, 100]
s_sv = [0.01, 0.1, 1, 10, 100]
for s_k in s_kv:
    for s_s in s_sv:
        Theta_init = np.random.multivariate_normal(np.zeros(m+1), np.eye(m+1))
        opt = minimize(class_log_nll_single, Theta_init, jac = class_log_nll_single_der, method='L-BFGS-B')
        Theta_k = opt['x'][0:m]
        b_k = opt['x'][m]
        
        yp = expit(xte @ Theta_k + b_k)
        yp[yp > 0.5] = 1
        yp[yp < 0.5] = 0
        tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
        acc_s = (tp + tn) / (tn + fp + fn + tp)
        hss2_s = (2 * ((tp * tn) - (fn*fp))) / ((tp + fn) * (fn + tn) + (tn + fp) * (tp + fp))
        acc_s_vec.append([acc_s, s_k, s_s])
    

x = [x, x_m]
y = [y, y_m]

acc_mt_vec = []
s_kv = [0.01, 0.1, 1, 10, 100]
s_sv = [0.01, 0.1, 1, 10, 100]
for s_k in s_kv:
    for s_s in s_sv:
        Theta_init = np.random.multivariate_normal(np.zeros(m+1), np.eye(m+1), size = K + 1)
        opt = minimize(class_log_nll_multi, Theta_init, jac = class_log_nll_multi_der,  method='L-BFGS-B')
        Theta_s = opt['x'][0:m]
        b_s = opt['x'][m]
        Theta_k = opt['x'][m+1:].reshape(K,m+1)[:, :m]
        b_k = opt['x'][m+1:].reshape(K,m+1)[:, m]
        
        yp = expit(xte @ Theta_k[0] + b_k[0])
        yp[yp > 0.5] = 1
        yp[yp < 0.5] = 0
        tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
        acc_mt = (tp + tn) / (tn + fp + fn + tp)
        hss2_mt = (2 * ((tp * tn) - (fn*fp))) / ((tp + fn) * (fn + tn) + (tn + fp) * (tp + fp))
        acc_mt_vec.append([acc_mt, s_k, s_s])