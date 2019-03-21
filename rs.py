import numpy as np
from sklearn import linear_model

N = MAX_USERS = 6040
M = MAX_MOVIES = 3952
K = 10
reg_const = 1.0
TEST_SIZE = 100076
#Lin = linear_model.Ridge(alpha=reg_const)

def read_data():
	num_nonzero = 0;
	data = np.zeros((MAX_USERS, MAX_MOVIES))
	Z = []
	for line in open('RSdata/training_rating.dat', 'r'):
		triple = line.split("::")
		if(triple[0] != '' and triple[1] != '' and triple[2] != ''):
			num_nonzero += 1
			uid = int(triple[0])
			mid = int(triple[1])
			rating = int(triple[2])
			data[uid-1, mid-1] = rating
			Z.append((uid-1, mid-1))
	return (data, Z, num_nonzero)

def get_test_indices():
	test_indices = np.zeros((TEST_SIZE, 2))
	count = 0
	for line in open('RSdata/testing.dat', 'r'):
		indices = line.split(' ')
		uid = int(indices[0])
		mid = int(indices[1])
		test_indices[count, 0] = uid
		test_indices[count, 1] = mid
		count += 1
	return test_indices

def make_cross_validation_data(R, Z, frac):
	train_size = int(frac * len(Z))
	test_size = len(Z) - train_size
	R_train = np.zeros((MAX_USERS, MAX_MOVIES))
	for i in range(train_size):
		(uid, mid) = Z[i]
		R_train[uid, mid] = R[uid, mid]
	Z_train = Z[:test_size]
	Z_test = Z[test_size:]
	return (R_train, Z_train, Z_test)

def calc_RMSE(U, V, R, test_indices):
	R_hat = U.dot(V.T)
	rmse = 0.0
	for k in range(len(test_indices)):
		(i, j) = test_indices[k]
		rmse += pow((max(0.0, R_hat[i, j]) - R[i, j]), 2)
	rmse = float(rmse) / float(len(test_indices))
	rmse = pow(rmse, 0.5)
	return rmse

def calc_loss(U, V, R, Z):
	R_hat = U.dot(V.T)
	error = 0.0
	for k in range(len(Z)):
		(i, j) = Z[k]
		if(R[i, j] != 0):
			error += pow(R[i, j] - max(0.0, R_hat[i, j]), 2)
			reg = pow(np.linalg.norm(U[i]), 2) + pow(np.linalg.norm(V[j]), 2)
			reg *= reg_const
			error += reg
			error *= 0.5
	return error

def update_U(init_U, init_V, R):
    U = np.zeros((N, K))
    for i in range(N):
        indices = np.nonzero(R[i])[0]
        if(len(indices) > 0):
            R_i = R[i][indices]
            V = init_V[indices, :]
            #Lin.fit(V, R_i)
            U[i] = np.linalg.lstsq(V.T.dot(V) + reg_const * np.identity(V.shape[1]), V.T.dot(R_i))[0]
    return U

def update_V(init_U, init_V, R):
    V = np.zeros((M, K))
    for j in range(M):
        indices = np.nonzero(R[:, j])[0]
        if(len(indices) > 0):
            R_j = R[indices, j]
            U = init_U[indices, :]
            #Lin.fit(U, R_j)
            V[j] = np.linalg.lstsq(U.T.dot(U) + reg_const * np.identity(U.shape[1]), U.T.dot(R_j))[0]
    return V

def als(U, V, R_train, Z_train, R=None, Z_test=None, max_epochs=100):
	num_epochs = 0
	loss = calc_loss(U, V, R_train, Z_train)
	rmse = calc_RMSE(U, V, R, Z_test)
	print('TRAINING')
	while(num_epochs <= max_epochs):
		print('loss: ' + str(loss) + ' num_epochs: ' + str(num_epochs) + ' rmse: ' + str(rmse))
		U = update_U(U, V, R_train)
		V = update_V(U, V, R_train)
		loss = calc_loss(U, V, R_train, Z_train)
		rmse = calc_RMSE(U, V, R, Z_test)
		num_epochs += 1
	final_loss = calc_loss(U, V, R_train, Z_train)
	return (U, V, final_loss)

def write_predictions(U, V, test_indices):
    R_hat = U.dot(V.T)
    predictions = np.zeros(TEST_SIZE)
    for i in range(TEST_SIZE):
        uid = int(test_indices[i, 0])
        mid = int(test_indices[i, 1])
        predictions[i] = max(0.0, R_hat[uid-1, mid-1])
    np.savetxt('results.csv', predictions, '%f')

print('Reading Data')
(R, Z, size) = read_data()
print('Getting Test Indices')
T = get_test_indices()
U = np.random.rand(N, K)
V = np.random.rand(M, K)
#print('Making Cross Validation Data')
#(R_train, Z_train, Z_test) = make_cross_validation_data(R, Z, 0.85)
print('Starting ALS')
(U, V, loss) = als(U, V, R, Z, R, Z)
print('Writing Predictions')
write_predictions(U, V, T)
print('Done')







