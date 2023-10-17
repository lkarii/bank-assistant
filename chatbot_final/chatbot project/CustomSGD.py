import numpy as np
import matplotlib.pyplot as plt
import copy
import time

if __name__ == '__main__':
    np.random.seed(1)
    lmd = 0.0001
    lr = 0.01
    num_iter = 200
    rho = 0.90

    # Data loading
    trainX = np.load('train_x_test.npy')
    trainY = np.load('train_y_test.npy')

    print('trainX shape:', trainX.shape)
    print('trainY shape:', trainY.shape)

    num_features = trainX.shape[1]
    W = np.random.normal(loc=0.0, scale=1.0, size=[trainX.shape[1], 1])
    W_old = copy.deepcopy(W)

    train_losses_sgdn = []
    train_accuracies_sgdn = []
    i = 0

    for iter in range(num_iter):
        train_loss_sum = 0
        batch_count = 0

        for x, y in zip(trainX, trainY):
            i += 1
            lr = 0.01 / i

            momentum = (1 + rho) * W - rho * W_old
            n_exp = np.exp(-y * np.matmul(x, momentum))

            exp = np.exp(-y * np.matmul(x, W))
            train_grad = -(np.expand_dims(x, axis=1).dot(np.expand_dims(y * n_exp / (1 + n_exp), axis=0))) / \
                         trainX.shape[0] + lmd * momentum

            W_old = copy.deepcopy(W)
            W = momentum - lr * train_grad

            train_loss = np.mean(np.log(1 + exp), axis=0) + (lmd * (np.sum(W ** 2)) / 2)
            train_loss_sum += train_loss
            batch_count += 1

        predictions = np.sign(np.matmul(trainX, W))
        accuracy = np.mean(predictions == trainY)
        if accuracy == 0:
            accuracy = np.finfo(float).eps
        train_accuracies_sgdn.append(accuracy)

        train_losses_sgdn.append(train_loss_sum / batch_count)
        print(iter, train_loss_sum / batch_count)

    plt.plot(np.arange(0, 200), np.log(train_losses_sgdn))
    plt.xlabel('Numer epoki')
    plt.ylabel('Strata/Dokładność')
    plt.title('Wartość funkcji straty i dokładności treningowej')
    plt.show()

    plt.plot(np.log(np.arange(0, 200), np.log(train_accuracies_sgdn)))
    plt.xlabel('Numer epoki')
    plt.ylabel('Strata/Dokładność')
    plt.title('Wartość funkcji straty i dokładności treningowej')
    plt.show()