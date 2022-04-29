import numpy as np
from test_utils import single_test, multiple_test

         
def initialize_parameters_test(target):
    n_x, n_h, n_y = 3, 2, 1
    expected_W1 = np.array([[ 0.01624345, -0.00611756, -0.00528172],
                     [-0.01072969,  0.00865408, -0.02301539]])
    expected_b1 = np.array([[0.],
                            [0.]])
    expected_W2 = np.array([[ 0.01744812, -0.00761207]])
    expected_b2 = np.array([[0.]])
    expected_output = {"W1": expected_W1,
                  "b1": expected_b1,
                  "W2": expected_W2,
                  "b2": expected_b2}
    test_cases = [
        {
            "name":"datatype_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error":"Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)
    
            
        
def initialize_parameters_deep_test(target):
    layer_dims = [5,4,3]
    expected_W1 = np.array([[ 0.01788628,  0.0043651 ,  0.00096497, -0.01863493, -0.00277388],
                        [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
                        [-0.01313865,  0.00884622,  0.00881318,  0.01709573,  0.00050034],
                        [-0.00404677, -0.0054536 , -0.01546477,  0.00982367, -0.01101068]])
    expected_b1 = np.array([[0.],
                            [0.],
                            [0.],
                            [0.]])
    expected_W2 = np.array([[-0.01185047, -0.0020565 ,  0.01486148,  0.00236716],
                         [-0.01023785, -0.00712993,  0.00625245, -0.00160513],
                         [-0.00768836, -0.00230031,  0.00745056,  0.01976111]])
    expected_b2 = np.array([[0.],
                            [0.],
                        [0.]])
    expected_output = {"W1": expected_W1,
                  "b1": expected_b1,
                  "W2": expected_W2,
                  "b2": expected_b2}
    test_cases = [
        {
            "name":"datatype_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)

def linear_forward_test(target):
    np.random.seed(1)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    expected_cache = (A_prev, W, b)
    expected_Z = np.array([[ 3.26295337, -1.23429987]])
    expected_output = (expected_Z, expected_cache)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b],
            "expected": expected_output,
            "error": "Wrong output"
        },
        
    ]
    
    multiple_test(test_cases, target)

def linear_activation_forward_test(target):
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    expected_linear_cache = (A_prev, W, b)
    expected_Z = np.array([[ 3.43896131, -2.08938436]])
    expected_cache = (expected_linear_cache, expected_Z)
    expected_A_sigmoid = np.array([[0.96890023, 0.11013289]])
    expected_A_relu = np.array([[3.43896131, 0.]])

    expected_output_sigmoid = (expected_A_sigmoid, expected_cache)
    expected_output_relu = (expected_A_relu, expected_cache)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [A_prev, W, b, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error":"Datatype mismatch with sigmoid activation"
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error": "Wrong shape with sigmoid activation"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error": "Wrong output with sigmoid activation"
        },
        {
            "name":"datatype_check",
            "input": [A_prev, W, b, 'relu'],
            "expected": expected_output_relu,
            "error":"Datatype mismatch with relu activation"
        },
        {
            "name": "shape_check",
            "input": [A_prev, W, b, 'relu'],
            "expected": expected_output_relu,
            "error": "Wrong shape with relu activation"
        },
        {
            "name": "equation_output_check",
            "input": [A_prev, W, b, 'relu'],
            "expected": expected_output_relu,
            "error": "Wrong output with relu activation"
        } 
    ]
    
    multiple_test(test_cases, target)    
        
def L_model_forward_test(target):
    np.random.seed(6)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
  
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    expected_cache = [((np.array([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
          [-2.48678065,  0.91325152,  1.12706373, -1.51409323],
          [ 1.63929108, -0.4298936 ,  2.63128056,  0.60182225],
          [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
          [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]]),
   np.array([[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
          [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
          [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
          [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]]),
   np.array([[ 1.38503523],
          [-0.51962709],
          [-0.78015214],
          [ 0.95560959]])),
  np.array([[-5.23825714,  3.18040136,  0.4074501 , -1.88612721],
         [-2.77358234, -0.56177316,  3.18141623, -0.99209432],
         [ 4.18500916, -1.78006909, -0.14502619,  2.72141638],
         [ 5.05850802, -1.25674082, -3.54566654,  3.82321852]])),
 ((np.array([[0.        , 3.18040136, 0.4074501 , 0.        ],
          [0.        , 0.        , 3.18141623, 0.        ],
          [4.18500916, 0.        , 0.        , 2.72141638],
          [5.05850802, 0.        , 0.        , 3.82321852]]),
   np.array([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
          [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
          [-0.37550472,  0.39636757, -0.47144628,  2.33660781]]),
   np.array([[ 1.50278553],
          [-0.59545972],
          [ 0.52834106]])),
  np.array([[ 2.2644603 ,  1.09971298, -2.90298027,  1.54036335],
         [ 6.33722569, -2.38116246, -4.11228806,  4.48582383],
         [10.37508342, -0.66591468,  1.63635185,  8.17870169]])),
 ((np.array([[ 2.2644603 ,  1.09971298,  0.        ,  1.54036335],
          [ 6.33722569,  0.        ,  0.        ,  4.48582383],
          [10.37508342,  0.        ,  1.63635185,  8.17870169]]),
   np.array([[ 0.9398248 ,  0.42628539, -0.75815703]]),
   np.array([[-0.16236698]])),
  np.array([[-3.19864676,  0.87117055, -1.40297864, -3.00319435]]))]
    expected_AL = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    expected_output = (expected_AL, expected_cache)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
'''        {
            "name":"datatype_check",
            "input": [AL, Y],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [AL, Y],
            "expected": expected_output,
            "error": "Wrong shape"
 },'''

def compute_cost_test(target):
    Y = np.asarray([[1, 1, 0]])
    AL = np.array([[.8,.9,0.4]])
    expected_output = np.array(0.27977656)

    test_cases = [
        {
            "name": "equation_output_check",
            "input": [AL, Y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)
    
def linear_backward_test(target):
    np.random.seed(1)
    dZ = np.random.randn(3,4)
    A = np.random.randn(5,4)
    W = np.random.randn(3,5)
    b = np.random.randn(3,1)
    linear_cache = (A, W, b)
    expected_dA_prev = np.array([[-1.15171336,  0.06718465, -0.3204696 ,  2.09812712],
       [ 0.60345879, -3.72508701,  5.81700741, -3.84326836],
       [-0.4319552 , -1.30987417,  1.72354705,  0.05070578],
       [-0.38981415,  0.60811244, -1.25938424,  1.47191593],
       [-2.52214926,  2.67882552, -0.67947465,  1.48119548]])
    expected_dW = np.array([[ 0.07313866, -0.0976715 , -0.87585828,  0.73763362,  0.00785716],
           [ 0.85508818,  0.37530413, -0.59912655,  0.71278189, -0.58931808],
           [ 0.97913304, -0.24376494, -0.08839671,  0.55151192, -0.10290907]])
    expected_db = np.array([[-0.14713786],
           [-0.11313155],
           [-0.13209101]])
    expected_output = (expected_dA_prev, expected_dW, expected_db)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [dZ, linear_cache],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [dZ, linear_cache],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [dZ, linear_cache],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
    
def linear_activation_backward_test(target):
    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    expected_dA_prev_sigmoid = np.array([[ 0.11017994,  0.01105339],
                             [ 0.09466817,  0.00949723],
                             [-0.05743092, -0.00576154]])
    expected_dW_sigmoid = np.array([[ 0.10266786,  0.09778551, -0.01968084]])
    expected_db_sigmoid = np.array([[-0.05729622]])
    expected_output_sigmoid = (expected_dA_prev_sigmoid, 
                               expected_dW_sigmoid, 
                               expected_db_sigmoid)
    
    expected_dA_prev_relu = np.array([[ 0.44090989,  0.        ],
           [ 0.37883606,  0.        ],
           [-0.2298228 ,  0.        ]])
    expected_dW_relu = np.array([[ 0.44513824,  0.37371418, -0.10478989]])
    expected_db_relu = np.array([[-0.20837892]])
    expected_output_relu = (expected_dA_prev_relu,
                           expected_dW_relu,
                           expected_db_relu)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [dA, linear_activation_cache, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error":"Data type mismatch with sigmoid activation"
        },
        {
            "name": "shape_check",
            "input": [dA, linear_activation_cache, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error": "Wrong shape with sigmoid activation"
        },
        {
            "name": "equation_output_check",
            "input": [dA, linear_activation_cache, 'sigmoid'],
            "expected": expected_output_sigmoid,
            "error": "Wrong output with sigmoid activation"
        } ,
        {
            "name":"datatype_check",
            "input": [dA, linear_activation_cache, 'relu'],
            "expected": expected_output_relu,
            "error":"Data type mismatch with relu activation"
        },
        {
            "name": "shape_check",
            "input": [dA, linear_activation_cache, 'relu'],
            "expected": expected_output_relu,
            "error": "Wrong shape with relu activation"
        },
        {
            "name": "equation_output_check",
            "input": [dA, linear_activation_cache, 'relu'],
            "expected": expected_output_relu,
            "error": "Wrong output with relu activation"
        }
    ]
    
    multiple_test(test_cases, target)
    
def L_model_backward_test(target):
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)
    
    expected_dA1 = np.array([[ 0.12913162, -0.44014127],
            [-0.14175655,  0.48317296],
            [ 0.01663708, -0.05670698]])
    expected_dW2 = np.array([[-0.39202432, -0.13325855, -0.04601089]])
    expected_db2 = np.array([[0.15187861]])
    expected_dA0 = np.array([[ 0.        ,  0.52257901],
            [ 0.        , -0.3269206 ],
            [ 0.        , -0.32070404],
            [ 0.        , -0.74079187]])
    expected_dW1 = np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
            [0.        , 0.        , 0.        , 0.        ],
            [0.05283652, 0.01005865, 0.01777766, 0.0135308 ]])
    expected_db1 = np.array([[-0.22007063],
            [ 0.        ],
            [-0.02835349]])
    expected_output = {'dA1': expected_dA1,
                       'dW2': expected_dW2,
                       'db2': expected_db2,
                       'dA0': expected_dA0,
                       'dW1': expected_dW1,
                       'db1': expected_db1
                      }
    test_cases = [
        {
            "name":"datatype_check",
            "input": [AL, Y, caches],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [AL, Y, caches],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [AL, Y, caches],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
    
def update_parameters_test(target):
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    learning_rate = 0.1
    expected_W1 = np.array([[-0.59562069, -0.09991781, -2.14584584,  1.82662008],
        [-1.76569676, -0.80627147,  0.51115557, -1.18258802],
        [-1.0535704 , -0.86128581,  0.68284052,  2.20374577]])
    expected_b1 = np.array([[-0.04659241],
            [-1.28888275],
            [ 0.53405496]])
    expected_W2 = np.array([[-0.55569196,  0.0354055 ,  1.32964895]])
    expected_b2 = np.array([[-0.84610769]])
    expected_output = {"W1": expected_W1,
                       'b1': expected_b1,
                       'W2': expected_W2,
                       'b2': expected_b2
                      }


    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, grads, learning_rate],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads, learning_rate],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads, 0.1],
            "expected": expected_output,
            "error": "Wrong output"
        }
        
    ]
    #print(target(*test_cases[2]["input"]))
    multiple_test(test_cases, target)

    
