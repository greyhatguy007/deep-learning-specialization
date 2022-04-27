import numpy as np
import copy
from test_utils import single_test, multiple_test
from testCases_v2 import nn_model_test_case

         
def layer_sizes_test(target):
    np.random.seed(1)
    X = np.random.randn(5, 3)
    Y = np.random.randn(2, 3)
    expected_output = (5, 4, 2)
    
    output = target(X, Y)
    
    assert type(output) == tuple, "Output must be a tuple"
    assert output == expected_output, f"Wrong result. Expected {expected_output} got {output}"
    
    X = np.random.randn(7, 5)
    Y = np.random.randn(5, 5)
    expected_output = (7, 4, 5)
    
    output = target(X, Y)
    
    assert type(output) == tuple, "Output must be a tuple"
    assert output == expected_output, f"Wrong result. Expected {expected_output} got {output}"
    
    print("\033[92mAll tests passed!")
            
        
def initialize_parameters_test(target):
    np.random.seed(2)
    n_x, n_h, n_y = 3, 5, 2

    expected_output = {'W1': np.array(
        [[-0.00416758, -0.00056267, -0.02136196],
         [ 0.01640271, -0.01793436, -0.00841747],
         [ 0.00502881, -0.01245288, -0.01057952],
         [-0.00909008,  0.00551454,  0.02292208],
         [ 0.00041539, -0.01117925,  0.00539058]]), 
                       'b1': np.array([[0.], [0.], [0.], [0.], [0.]]), 
                       'W2': np.array([[-5.96159700e-03, -1.91304965e-04,  1.17500122e-02,
        -7.47870949e-03,  9.02525097e-05],
       [-8.78107893e-03, -1.56434170e-03,  2.56570452e-03,
        -9.88779049e-03, -3.38821966e-03]]), 
                       'b2': np.array([[0.], [0.]])}
    
    parameters = target(n_x, n_h, n_y)
    
    assert type(parameters["W1"]) == np.ndarray, f"Wrong type for W1. Expected: {np.ndarray}"
    assert type(parameters["b1"]) == np.ndarray, f"Wrong type for b1. Expected: {np.ndarray}"
    assert type(parameters["W2"]) == np.ndarray, f"Wrong type for W2. Expected: {np.ndarray}"
    assert type(parameters["b2"]) == np.ndarray, f"Wrong type for b2. Expected: {np.ndarray}"
    
    assert parameters["W1"].shape == expected_output["W1"].shape, f"Wrong shape for W1."
    assert parameters["b1"].shape == expected_output["b1"].shape, f"Wrong shape for b1."
    assert parameters["W2"].shape == expected_output["W2"].shape, f"Wrong shape for W2."
    assert parameters["b2"].shape == expected_output["b2"].shape, f"Wrong shape for b2."
    
    assert np.allclose(parameters["W1"], expected_output["W1"]), "Wrong values for W1"
    assert np.allclose(parameters["b1"], expected_output["b1"]), "Wrong values for b1"
    assert np.allclose(parameters["W2"], expected_output["W2"]), "Wrong values for W2"
    assert np.allclose(parameters["b2"], expected_output["b2"]), "Wrong values for b2"
   
    print("\033[92mAll tests passed!")


def forward_propagation_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    b1 = np.random.randn(4, 1)
    b2 = np.array([[ -1.3]]) 

    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': b1,
     'b2': b2}
    expected_A1 = np.array([[ 0.9400694 ,  0.94101876,  0.94118266],
                     [-0.67151964, -0.62547205, -0.65709025],
                     [ 0.29034152,  0.31196971,  0.33449821],
                     [-0.22397799, -0.25730819, -0.2197236 ]])
    expected_A2 = np.array([[0.21292656, 0.21274673, 0.21295976]])

    expected_Z1 = np.array([[ 1.7386459 ,  1.74687437,  1.74830797],
                        [-0.81350569, -0.73394355, -0.78767559],
                        [ 0.29893918,  0.32272601,  0.34788465],
                        [-0.2278403 , -0.2632236 , -0.22336567]])

    expected_Z2 = np.array([[-1.30737426, -1.30844761, -1.30717618]])
    expected_cache = {"Z1": expected_Z1,
             "A1": expected_A1,
             "Z2": expected_Z2,
             "A2": expected_A2}
    expected_output = (expected_A2, expected_cache)
    
    output = target(X, parameters)
    
    assert type(output[0]) == np.ndarray, f"Wrong type for A2. Expected: {np.ndarray}"
    assert type(output[1]["Z1"]) == np.ndarray, f"Wrong type for cache['Z1']. Expected: {np.ndarray}"
    assert type(output[1]["A1"]) == np.ndarray, f"Wrong type for cache['A1']. Expected: {np.ndarray}"
    assert type(output[1]["Z2"]) == np.ndarray, f"Wrong type for cache['Z2']. Expected: {np.ndarray}"
    
    assert output[0].shape == expected_A2.shape, f"Wrong shape for A2."
    assert output[1]["Z1"].shape ==expected_Z1.shape, f"Wrong shape for cache['Z1']."
    assert output[1]["A1"].shape == expected_A1.shape, f"Wrong shape for cache['A1']."
    assert output[1]["Z2"].shape == expected_Z2.shape, f"Wrong shape for cache['Z2']."
    
    assert np.allclose(output[0], expected_A2), "Wrong values for A2"
    assert np.allclose(output[1]["Z1"], expected_Z1), "Wrong values for cache['Z1']"
    assert np.allclose(output[1]["A1"], expected_A1), "Wrong values for cache['A1']"
    assert np.allclose(output[1]["Z2"], expected_Z2), "Wrong values for cache['Z2']"
    
    print("\033[92mAll tests passed!")
    

def compute_cost_test(target):
    np.random.seed(1)
    Y = (np.random.randn(1, 5) > 0)
    A2 = (np.array([[ 0.5002307 ,  0.49985831,  0.50023963, 0.25, 0.7]]))

    expected_output = 0.5447066599017815
    output = target(A2, Y)
    
    assert type(output) == float, "Wrong type. Float expected"
    assert np.isclose(output, expected_output), f"Wrong value. Expected: {expected_output} got: {output}"
  
    print("\033[92mAll tests passed!")
        
def backward_propagation_test(target):
    np.random.seed(1)
    X = np.random.randn(3, 7)
    Y = (np.random.randn(1, 7) > 0)
    parameters = {'W1': np.random.randn(9, 3),
         'W2': np.random.randn(1, 9),
         'b1': np.array([[ 0.], [ 0.], [ 0.], [ 0.], [ 0.], [ 0.], [ 0.], [ 0.], [ 0.]]),
         'b2': np.array([[ 0.]])}

    cache = {'A1': np.random.randn(9, 7),
         'A2': np.random.randn(1, 7),
         'Z1': np.random.randn(9, 7),
         'Z2': np.random.randn(1, 7),}

    
    expected_output = {'dW1': np.array([[-0.24468458, -0.24371232,  0.15959609],
                        [ 0.7370069 , -0.64785999,  0.23669823],
                        [ 0.47936123, -0.01516428,  0.01566728],
                        [ 0.03361075, -0.0930929 ,  0.05581073],
                        [ 0.52445178, -0.03895358,  0.09180612],
                        [-0.17043596,  0.13406378, -0.20952062],
                        [ 0.76144791, -0.41766018,  0.02544078],
                        [ 0.22164791, -0.33081645,  0.19526915],
                        [ 0.25619969, -0.09561825,  0.05679075]]),
                 'db1': np.array([[ 0.1463639 ],
                        [-0.33647992],
                        [-0.51738006],
                        [-0.07780329],
                        [-0.57889514],
                        [ 0.28357278],
                        [-0.39756864],
                        [-0.10510329],
                        [-0.13443244]]),
                 'dW2': np.array([[-0.35768529,  0.22046323, -0.29551566, -0.12202786,  0.18809552,
                          0.13700323,  0.35892872, -0.02220353, -0.03976687]]),
                 'db2': np.array([[-0.78032466]])}
    
    output = target(parameters, cache, X, Y)
    
    assert type(output["dW1"]) == np.ndarray, f"Wrong type for dW1. Expected: {np.ndarray}"
    assert type(output["db1"]) == np.ndarray, f"Wrong type for db1. Expected: {np.ndarray}"
    assert type(output["dW2"]) == np.ndarray, f"Wrong type for dW2. Expected: {np.ndarray}"
    assert type(output["db2"]) == np.ndarray, f"Wrong type for db2. Expected: {np.ndarray}"
    
    assert output["dW1"].shape == expected_output["dW1"].shape, f"Wrong shape for dW1."
    assert output["db1"].shape == expected_output["db1"].shape, f"Wrong shape for db1."
    assert output["dW2"].shape == expected_output["dW2"].shape, f"Wrong shape for dW2."
    assert output["db2"].shape == expected_output["db2"].shape, f"Wrong shape for db2."
    
    assert np.allclose(output["dW1"], expected_output["dW1"]), "Wrong values for dW1"
    assert np.allclose(output["db1"], expected_output["db1"]), "Wrong values for db1"
    assert np.allclose(output["dW2"], expected_output["dW2"]), "Wrong values for dW2"
    assert np.allclose(output["db2"], expected_output["db2"]), "Wrong values for db2"
    
    print("\033[92mAll tests passed!")


def update_parameters_test(target):
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
 'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
 'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
 'b2': np.array([[  9.14954378e-05]])}

    grads = {'dW1': np.array([[ 0.00023322, -0.00205423],
        [ 0.00082222, -0.00700776],
        [-0.00031831,  0.0028636 ],
        [-0.00092857,  0.00809933]]),
 'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03,
          -2.55715317e-03]]),
 'db1': np.array([[  1.05570087e-07],
        [ -3.81814487e-06],
        [ -1.90155145e-07],
        [  5.46467802e-07]]),
 'db2': np.array([[ -1.08923140e-05]])}
    
    expected_W1 = np.array([[-0.00643025,  0.01936718],
        [-0.02410458,  0.03978052],
        [-0.01653973, -0.02096177],
        [ 0.01046864, -0.05990141]])
    expected_b1 = np.array([[-1.02420756e-06],
            [ 1.27373948e-05],
            [ 8.32996807e-07],
            [-3.20136836e-06]])
    expected_W2 = np.array([[-0.01041081, -0.04463285,  0.01758031,  0.04747113]])
    expected_b2 = np.array([[0.00010457]])
    
    expected_output = {"W1": expected_W1,
                  "b1": expected_b1,
                  "W2": expected_W2,
                  "b2": expected_b2}
    
    output = target(parameters, grads)

    assert type(output["W1"]) == np.ndarray, f"Wrong type for W1. Expected: {np.ndarray}"
    assert type(output["b1"]) == np.ndarray, f"Wrong type for b1. Expected: {np.ndarray}"
    assert type(output["W2"]) == np.ndarray, f"Wrong type for W2. Expected: {np.ndarray}"
    assert type(output["b2"]) == np.ndarray, f"Wrong type for b2. Expected: {np.ndarray}"
    
    assert output["W1"].shape == expected_output["W1"].shape, f"Wrong shape for W1."
    assert output["b1"].shape == expected_output["b1"].shape, f"Wrong shape for b1."
    assert output["W2"].shape == expected_output["W2"].shape, f"Wrong shape for W2."
    assert output["b2"].shape == expected_output["b2"].shape, f"Wrong shape for b2."
    
    assert np.allclose(output["W1"], expected_output["W1"]), "Wrong values for W1"
    assert np.allclose(output["b1"], expected_output["b1"]), "Wrong values for b1"
    assert np.allclose(output["W2"], expected_output["W2"]), "Wrong values for W2"
    assert np.allclose(output["b2"], expected_output["b2"]), "Wrong values for b2"
    
    print("\033[92mAll tests passed!")
    
def nn_model_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = (np.random.randn(1, 3) > 0)
    n_h = 4
    
    t_X, t_Y = nn_model_test_case()
    parameters = target(t_X, t_Y, n_h, num_iterations=10000, print_cost=True)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    expected_output = {'W1': np.array([[ 0.56305445, -1.03925886],
                                   [ 0.7345426 , -1.36286875],
                                   [-0.72533346,  1.33753027],
                                   [ 0.74757629, -1.38274074]]), 
                       'b1': np.array([[-0.22240654],
                                   [-0.34662093],
                                   [ 0.33663708],
                                   [-0.35296113]]), 
                       'W2': np.array([[ 1.82196893,  3.09657075, -2.98193564,  3.19946508]]), 
                       'b2': np.array([[0.21344644]])}
    
    np.random.seed(3)
    output = target(X, Y, n_h, print_cost=False)
    
    assert type(output["W1"]) == np.ndarray, f"Wrong type for W1. Expected: {np.ndarray}"
    assert type(output["b1"]) == np.ndarray, f"Wrong type for b1. Expected: {np.ndarray}"
    assert type(output["W2"]) == np.ndarray, f"Wrong type for W2. Expected: {np.ndarray}"
    assert type(output["b2"]) == np.ndarray, f"Wrong type for b2. Expected: {np.ndarray}"
    
    assert output["W1"].shape == expected_output["W1"].shape, f"Wrong shape for W1."
    assert output["b1"].shape == expected_output["b1"].shape, f"Wrong shape for b1."
    assert output["W2"].shape == expected_output["W2"].shape, f"Wrong shape for W2."
    assert output["b2"].shape == expected_output["b2"].shape, f"Wrong shape for b2."
    
    assert np.allclose(output["W1"], expected_output["W1"]), "Wrong values for W1"
    assert np.allclose(output["b1"], expected_output["b1"]), "Wrong values for b1"
    assert np.allclose(output["W2"], expected_output["W2"]), "Wrong values for W2"
    assert np.allclose(output["b2"], expected_output["b2"]), "Wrong values for b2"
    
    print("\033[92mAll tests passed!")

    
def predict_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    expected_output = np.array([[True, False, True]])
    
    output = target(parameters, X)
    
    assert np.array_equal(output, expected_output), f"Wrong prediction. Expected: {expected_output} got: {output}"
    
    print("\033[92mAll tests passed!")


    
