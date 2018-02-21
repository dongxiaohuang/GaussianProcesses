import numpy as np
from scipy.optimize import minimize
# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    sample = np.zeros((mean.shape[0], )) # This is only a placeholder
    # Task 1:
    # TODO: Implement a draw from a multivariate Gaussian here
    A = np.linalg.cholesky(cov)
    x = np.random.randn(mean.shape[0],)
    sample = np.dot(A, x.T) + mean
    # Return drawn sample
    return sample

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug

        n = X.shape[0]
        covMat = np.zeros((n,n))

        # Task 2:
        # TODO: Implement the covariance matrix here
        for p in range(n):
            for q in range(n):
                covMat[p][q] = self.sigma2_f*np.exp(-1.0/(2*self.length_scale*self.length_scale)*((np.linalg.norm(X[p]-X[q]))**2))

        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        return K

    # ############################    cov_fa = covMatrix(Xa)##############################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ############################    cov_fa = covMatrix(Xa)##############################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        print Xa
        mean_fa = (Xa.sum(axis = 1)/len(Xa[0])).reshape(mean_fa.shape)
        for i in range(Xa.shape[0]):
            for j in range(Xa.shape[0]):
                E_ij = 0;
                for p in Xa[i]:
                    for q in Xa[j]:
                        E_ij += p*q
                E_ij /= len(Xa[i])*len(Xa[j])
                cov_fa[i][j] = E_ij - mean_fa[i]*mean_fa[j]

        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)
            #TODO: add
        mll = 0
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y

        det_K = np.linalg.det(self.K)
        sub1 = 0.5*self.y.T.dot(np.linalg.inv(self.K))
        sub2 = sub1.dot(self.y)
        mll = sub2 + .5*np.log(det_K)+.5*self.n*np.log(2*np.pi)
        # Return mll
        return mll

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters
        n = (self.K).shape[0]
        alph = np.linalg.inv(self.K).dot(self.y)
        if self.k.sigma2_n is not None:
            K_wo_noise = self.K - self.k.sigma2_n*np.identity(n)
        grad_ln_sigma_f = 0.5* np.trace(
        (alph.dot(alph.T) - np.linalg.inv(self.K)).dot(2*K_wo_noise)
        )
        grad_ln_length_scale =0.5* np.trace(
        (alph.dot(alph.T) - np.linalg.inv(self.K)).dot(2*self.k.sigma2_n*np.identity(n))
        )

        # grad_ln_sigma_n =

        # Combine gradients
        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        n = len(ya)
        for i in range(n):
            mse += ((ya[i]-fbar[i])**2)
        mse /= n;
        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        n = len(ya)
        pi = np.pi
        #TODO: signma2_xa
        for i in range(n):
            sigma2_xi = cov[i][i] + self.k.sigma2_n
            msll += .5 * np.log(2*pi*sigma2_xi) + 0.5*((ya[i] - fbar[i])**2) / (sigma2_xi)
        msll /= n
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        return res.x

if __name__ == '__main__':

    np.random.seed(42)
    # params = (0.5*math.log(1.0), math.log(0.1), 0.5* math.log(0.5))
    # gpg = GaussianProcessRegression().optimize()
    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################
    X_train, y_train, X_test, y_test = loadData("yacht_hydrodynamics.data")
    RBF = RadialBasisFunction([0.0, np.log(.1), .5*np.log(0.5)])
    gpr = GaussianProcessRegression(X_train, y_train, RBF)

    gpr.predict(X_test)
