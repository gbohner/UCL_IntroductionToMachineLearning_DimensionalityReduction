import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy


def custom_lle(X, n_neighbours=12, k=2, regularization=1e-3):
    #X: raw input data
    
    #Step 1 - For each point get the local neighbourhood based on pairwise distance measures
    # Input - data X (features x samples) and n_neighbors. 
    # Output - local neighbourhood indices (n_neighbours x samples)

    D, N = X.shape #D - number of features, N - number of samples
    
    def euc_dist(X):
        #Get euclidean pairwise distances
    	Xdist = np.zeros((X.shape[1], X.shape[1]))
    	for i in range(X.shape[1]):
    		Xdist[i,:] = np.sqrt(np.sum((X - X[:,i][:,np.newaxis])**2, axis=0))    
    	return Xdist #Each i,j element contains the euclidean distance between the i-th and j-th column of X

    def get_neighborhoods(Xdist, n_neighbours):
    	# Get the neighborhood of each point based on the euclidean distances
    	return np.argsort(Xdist,axis=0)[1:n_neighbours+1,:] #The zeroth element is always going to be the point itself, we want the first n_neighbourhood afterwards
    

    Xneigh = get_neighborhoods(euc_dist(X), n_neighbours)


    #Step 2 - For each point solve reconstruction using it's neighbourhood as regressors
    # Input - data,  neighbourhoods. 
    # Output - local reconstruction weights W (n_neighbours x samples)
    
    # Less intuitive but numerically more stable solution:
    # W = np.zeros((n_neighbours, N))
    # for i in range(N):
    #     A = X[:,Xneigh[:,i]] - X[:,i][:,np.newaxis]
    #     C = A.T.dot(A)
    #     C = C + regularization * np.eye(n_neighbours) * np.trace(C)
    #     W[:,i] = np.linalg.solve(C, np.ones(n_neighbours))
    #     W[:,i] = W[:,i]/np.sum(W[:,i])
    # This W is the very same as in the Matlab version
                                    
    # Equivalent solution with more intuition and less numerical stability
    W = np.zeros((n_neighbours, N))                          
    for i in range(N):
    	# For each neighborhood, solve the local regularized linear regression with constraint that sum(w)=1
    	A = X[:,Xneigh[:,i]] # Get the regressors, solve min_w ||A w - X[:,i]||^2 + regularization*||w||^2, with w as the cofficients (sum(w)=1 constraint)
    	# General regularized regression solution (you might wanna learn this well... )        
    	W[:,i] = np.linalg.inv(A.T.dot(A) + regularization*np.eye(n_neighbours)).dot(A.T).dot(X[:,i])
        # This is the solution for the unconstrained problem.
        # However our constraint is that W[:,i] lies on the unit ball. This is easy to fulfil, as we 
        # have to find the closest point on the unit ball to the unconstrained solution, which is simply normalizing the vector
    	W[:,i] = W[:,i]/np.sum(W[:,i])

        # Slight numerical differences between the two solutions, mainly due to linalg.inv being terrible without regularization 
        # (should use "solve" in general, but formula is more understandable this way)


    #Step 3 - Compute the best k-dimensional embedding Y by solving the quadratic cost function
    # Input - W
    # Output - Y (k x samples)
    # Note that you do not get "projection vectors" as in PCA, only the projected data
    # This is due to the fact that globally this method is non-linear.


    # Conceptually this part is somewhat difficult
    # We want to find Y, a (k x samples) matrix, such that these embeddings minimize the cost function
    # Cost(Y) = sum_i || Y[:,i] - W[:,i].T.dot(Y[:,Xneigh[:,i]].T) ) ||^2
	# subject to some constraints (see lecture notes)
    # Meaning we retain as much of the local "reconstructability" by the neighbours as possible

    # Writing down a "design matrix" M, such that Phi = ( I - W).T.dot(I - W), the cost function simplifies
    # Cost(Y) = sum(sum( Phi * (Y.T.dot(Y)) )) , where * is elementwise product

    # This is analytically solveable by setting Y to the bottom 2 ... k+1 eigenvectors of Phi
    
    #Phi = (np.eye(N) - W).T.dot(np.eye(N) - W)
    #d,Y = np.linalg.eig(Phi)

    Phi = np.eye(N)
    
    # Filling up Phi as Phi = I - W - W^T + WxW, where W now is the full sparse weight matrix (with elements non-zero where there is a neighbourhood)
    for i in range(N):
        ne = Xneigh[:,i]
        w = W[:,i]
        Phi[i,ne] = Phi[i,ne] - w.T
        Phi[ne,i] = Phi[ne,i] - w
        Phi[np.ix_(ne,ne)] = Phi[np.ix_(ne,ne)] + np.outer(w,w)
    
    d,Y = np.linalg.eigh(Phi)
    order = np.argsort(d) # ascending sort (lowest eigenvalues first)
    Y = Y[:,order][:,k:0:-1] # Python indexing is a bit weird, this actually is the k+1-th, k-th, ... , 2nd eigenvector belonging to the respective smallest eigenvalues

    
    return Y #W, Phi, Xneigh # - if you wanna look at other quantities


# Further solutions


# PCA correct code
# # Correct implementation
# N = X.shape[1]
# # Substract the mean from the data
# Xctr = X - X.mean(axis=1)[:,np.newaxis]
# Dctr,Vctr = np.linalg.eig(np.dot(Xctr,Xctr.T)/N) # Get eigenvalues and eigenvectors (unsorted)

# Helper function to visualize image arrays
def showfreyface(X, ndims=0, scale=1, figtitle=0):	
	sz = X.shape
	if ndims == 0:
		num_cols = int(np.ceil(np.sqrt(sz[1]*28.0/20.0 * 10/8)))
		num_rows = int(np.floor(sz[1]/num_cols) + (np.mod(sz[1],num_cols)>0))
	else:
		assert np.prod(ndims[0:2])>=sz[1], 'You want to display more images than provided grid size'
		num_cols = int(ndims[0])
		num_rows = int(ndims[1])

	matplotlib.rcParams['figure.figsize'] = (10.0*scale, 8.0*scale)
	fig = plt.figure()
	if not figtitle==0:
		fig.suptitle(figtitle)
	gs = matplotlib.gridspec.GridSpec(num_rows, num_cols, wspace=0.05, hspace=0.05)

	ax = [plt.subplot(gs[i]) for i in range(num_rows*num_cols)]

	for i in range(num_rows*num_cols):
	    if i<sz[1]:
	        ax[i].imshow(X[:,i].reshape((28,20)), cmap='gray')
	    ax[i].axis('off')

	plt.show()






