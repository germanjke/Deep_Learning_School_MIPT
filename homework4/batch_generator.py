def batch_generator(X, y, batch_size):
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    
    # YOUR CODE
    num_batches = math.ceil(len(X)/batch_size) # кол-во батчей, округляем до целого сверху
    for i in range(num_batches): 
        if i == (num_batches - 1): #случай для последнего батча, заполняем всем, что есть
            batch_indexes = perm[i*batch_size:]
        else: #случай для всех батчей, кроме последнего, заполняем от края до края
            batch_indexes = perm[i*batch_size:batch_size*(i+1)] 
        
        X_batch = X[batch_indexes] #заполняем батчи
        y_batch = y[batch_indexes] #заполняем батчи
        
        yield X_batch, y_batch
