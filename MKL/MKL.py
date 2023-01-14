class GreedyKernel:
    """
    Class for greedy growing kernel structure
    """
    def __init__(self, algebra, base_kernels):
        self.algebra = algebra
        self.base_kernels = base_kernels
        self.bic = {}
        self.kernel = None
        self.kernel_list = []
        self.op_list = []
        self.str_kernel = None
        self.runtime = {}
        self.loss = {}
        self.mll = None
    
    def _make_kernel(self, op_list, kernel_list):
        """
        Create kernel according to operation and kernel lists
        """
        kernels_to_sum = _get_all_product_kernels(op_list, kernel_list)
        new_kernel = kernels_to_sum[0]
        for k in kernels_to_sum[1:]:
            new_kernel = new_kernel + k
        return new_kernel
    
    def init_kernel(self, X_train, y_train):
        """
        Find initial single best kernel
        """
        best_kernel = None
        
        bic = np.zeros(len(self.base_kernels))
        
        for k, kernel in enumerate(self.base_kernels):
            start = time.time()
            bic[k], ls = train_model_get_bic(X_train, y_train, kernel)
            end = time.time()
            runtime = end - start
            kernel_name = str(kernel.base_kernel).split('(')[0]
            self.loss[kernel_name] = ls
            self.bic[kernel_name] = bic
            self.runtime[kernel_name] = runtime
                
        best_kernel = self.base_kernels[bic.argmin()]
                
        assert best_kernel is not None
        
        self.kernel_list.append(best_kernel)
        self.str_kernel = str(best_kernel.base_kernel).split('(')[0]
        
    def grow_level(self, X_train, y_train):
        """
        Select optimal extension of current kernel (add one new kernel)
        """
        
        best_kernel = None  # should be kernel object
        best_op = None  # should be operation name, i.e. "+" or "*"
        
        # base kernels are given by self.base_kernels --- list of kernel objects
        # operations are given by self.algebra --- dictionary:
        #                                              {"+": lambda x, y: x + y
        #                                               "*": lambda x, y: x * y}

        # best_kernel - kernel object, store in this variable the best found kernel
        # best_op - '+' or '*', store in this variable the best found operation
        
        best_bic = np.inf
        for k, kernel in enumerate(self.base_kernels):
            for a, (op_name, op) in enumerate(self.algebra.items()):
                start = time.time()
                new_kernel = self._make_kernel(self.op_list + [op_name], self.kernel_list + [kernel])
                bic, ls = train_model_get_bic(X_train, y_train, new_kernel)
                end = time.time()
                runtime = end - start
                kernel_name = '{} {} {}'.format(self.str_kernel, op_name,
                                           str(kernel.base_kernel).split('(')[0])

                self.loss[kernel_name] = ls
                self.bic[kernel_name] = bic
                self.runtime[kernel_name] = runtime
                
                if bic < best_bic:
                    best_op = op_name
                    best_kernel = kernel
                    best_bic = bic
                
        self.kernel_list.append(copy.deepcopy(best_kernel))
        self.op_list.append(best_op)
        
        assert best_kernel is not None
        assert best_op is not None
        
        self.kernel_list.append(best_kernel)
        self.op_list.append(best_op)
        
        new_kernel = self._make_kernel(self.op_list, self.kernel_list)
        str_new_kernel = '{} {} {}'.format(self.str_kernel, best_op,
                                           str(best_kernel.base_kernel).split('(')[0])
        
        return new_kernel, str_new_kernel
    
    def grow_tree(self, X_train, y_train, max_depth):
        """
        Greedy kernel construction
        """
        if self.kernel == None:
            self.init_kernel(X_train, y_train)
            
        for i in range(max_depth):
            self.kernel, self.str_kernel = self.grow_level(X_train, y_train)
            print(self.str_kernel)
        return self.kernel_list
            
    def fit_model(self, X_train, y_train, kernel, n_epochs):
        start = time.time()
        model = MTGPRegressor(X_train, y_train, kernel)
        ls, self.mll = training(model, X_train, y_train, n_epochs=n_epochs)
        end = time.time()
        runtime = end - start
        
        self.loss[self.str_kernel] = ls
        self.runtime[self.str_kernel] = runtime
        
        return model