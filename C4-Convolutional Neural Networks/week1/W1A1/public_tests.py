import numpy as np
from outputs import *


def zero_pad_test(target):    
    # Test 1
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = target(x, 3)
    print ("x.shape =\n", x.shape)
    print ("x_pad.shape =\n", x_pad.shape)
    print ("x[1,1] =\n", x[1, 1])
    print ("x_pad[1,1] =\n", x_pad[1, 1])

    assert type(x_pad) == np.ndarray, "Output must be a np array"
    assert x_pad.shape == (4, 9, 9, 2), f"Wrong shape: {x_pad.shape} != (4, 9, 9, 2)"
    print(x_pad[0, 0:2,:, 0])
    assert np.allclose(x_pad[0, 0:2,:, 0], [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "Rows are not padded with zeros"
    assert np.allclose(x_pad[0, :, 7:9, 1].transpose(), [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "Columns are not padded with zeros"
    assert np.allclose(x_pad[:, 3:6, 3:6, :], x, 1e-15), "Internal values are different"

    # Test 2
    np.random.seed(1)
    x = np.random.randn(5, 4, 4, 3)
    pad = 2
    x_pad = target(x, pad)
    
    assert type(x_pad) == np.ndarray, "Output must be a np array"
    assert x_pad.shape == (5, 4 + 2 * pad, 4 + 2 * pad, 3), f"Wrong shape: {x_pad.shape} != {(5, 4 + 2 * pad, 4 + 2 * pad, 3)}"
    assert np.allclose(x_pad[0, 0:2,:, 0], [[0, 0, 0, 0, 0, 0, 0, 0], 
                                            [0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "Rows are not padded with zeros"
    assert np.allclose(x_pad[0, :, 6:8, 1].transpose(), [[0, 0, 0, 0, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 0, 0, 0, 0]], 1e-15), "Columns are not padded with zeros"
    assert np.allclose(x_pad[:, 2:6, 2:6, :], x, 1e-15), "Internal values are different"
    
    print("\033[92mAll tests passed!")
    


def conv_single_step_test(target):

    np.random.seed(3)
    a_slice_prev = np.random.randn(5, 5, 3)
    W = np.random.randn(5, 5, 3)
    b = np.random.randn(1, 1, 1)
    
    Z = target(a_slice_prev, W, b)
    expected_output = np.float64(-3.5443670581382474)
    
    assert (type(Z) == np.float64 or type(Z) == np.float32), "You must cast the output to float"
    assert np.isclose(Z, expected_output), f"Wrong value. Expected: {expected_output} got: {Z}"
    
    print("\033[92mAll tests passed!")

def conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3):
    test_count = 0
    z_mean_expected = 0.5511276474566768
    z_0_2_1_expected = [-2.17796037,  8.07171329, -0.5772704,   3.36286738,  4.48113645, -2.89198428, 10.99288867,  3.03171932]
    cache_0_1_2_3_expected = [-1.1191154,   1.9560789,  -0.3264995,  -1.34267579]
    
    if np.isclose(z_mean, z_mean_expected):
        test_count = test_count + 1
    else:
        print("\033[91mFirst Test: Z's mean is incorrect. Expected:", z_mean_expected, "\nYour output:", z_mean, "\033[90m\n")
        
    if np.allclose(z_0_2_1, z_0_2_1_expected):
        test_count = test_count + 1
    else:
        print("\033[91mFirst Test: Z[0,2,1] is incorrect. Expected:", z_0_2_1_expected, "\nYour output:", z_0_2_1, "\033[90m\n")
        
    if np.allclose(cache_0_1_2_3, cache_0_1_2_3_expected):
        test_count = test_count + 1
    else:
        print("\033[91mFirst Test: cache_conv[0][1][2][3] is incorrect. Expected:", cache_0_1_2_3_expected, "\nYour output:",
              cache_0_1_2_3, "\033[90m")
    
    if test_count == 3:
        print("\033[92mFirst Test: All tests passed!")
    
def conv_forward_test_2(target):
    # Test 1
    np.random.seed(3)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    
    Z, cache_conv = target(A_prev, W, b, {"pad" : 3, "stride": 1})
    Z_shape = Z.shape
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 9, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 9"
    assert Z_shape[2] == 11, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 11"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"

    # Test 2 
    Z, cache_conv = target(A_prev, W, b, {"pad" : 0, "stride": 2})
    assert(Z.shape == (2, 2, 3, 8)), "Wrong shape. Don't hard code the pad and stride values in the function"
    
    # Test 3
    W = np.random.randn(5, 5, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    Z, cache_conv = target(A_prev, W, b, {"pad" : 6, "stride": 1})
    Z_shape = Z.shape
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 13, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 13"
    assert Z_shape[2] == 15, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 15"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"

    Z_means = np.mean(Z)
    expected_Z = -0.5384027772160062
    
    expected_conv = np.array([[ 1.98848968,  1.19505834, -0.0952376,  -0.52718778],
                             [-0.32158469,  0.15113037, -0.01862772,  0.48352879],
                             [ 0.76896516,  1.36624284,  1.14726479, -0.11022916],
                             [ 0.38825041, -0.38712718, -0.58722031,  1.91082685],
                             [-0.45984615,  1.99073781, -0.34903539,  0.25282509],
                             [ 1.08940955,  0.02392202,  0.39312528, -0.2413848 ],
                             [-0.47552486, -0.16577702, -0.64971742,  1.63138295]])
    
    assert np.isclose(Z_means, expected_Z), f"Wrong Z mean. Expected: {expected_Z} got: {Z_means}"
    assert np.allclose(cache_conv[0][1, 2], expected_conv), f"Values in Z are wrong"

    print("\033[92mSecond Test: All tests passed!")


def pool_forward_test(target):
    
    # Test 1
    A_prev = np.random.randn(2, 5, 7, 3)
    A, cache = target(A_prev, {"stride" : 2, "f": 2}, mode = "average")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"Test 1 - m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 2, f"Test 1 - n_H is wrong. Current: {A_shape[1]}.  Expected: 2"
    assert A_shape[2] == 3, f"Test 1 - n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"Test 1 - n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"
    
    # Test 2
    A_prev = np.random.randn(4, 5, 7, 4)
    A, cache = target(A_prev, {"stride" : 1, "f": 5}, mode = "max")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"Test 2 - m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 1, f"Test 2 - n_H is wrong. Current: {A_shape[1]}.  Expected: 1"
    assert A_shape[2] == 3, f"Test 2 - n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"Test 2 - n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"
    
    # Test 3
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)
    
    A, cache = target(A_prev, {"stride" : 1, "f": 2}, mode = "max")
    
    assert np.allclose(A[1, 1], np.array([[1.19891788, 0.74055645, 0.07734007],
                                         [0.31515939, 0.84616065, 0.07734007],
                                         [0.69803203, 0.84616065, 1.2245077 ],
                                         [0.69803203, 1.12141771, 1.2245077 ]])), "Wrong value for A[1, 1]"
                                          
    assert np.allclose(cache[0][1, 2], np.array([[ 0.16938243,  0.74055645, -0.9537006 ],
                                         [-0.26621851,  0.03261455, -1.37311732],
                                         [ 0.31515939,  0.84616065, -0.85951594],
                                         [ 0.35054598, -1.31228341, -0.03869551],
                                         [-1.61577235,  1.12141771,  0.40890054]])), "Wrong value for cache"
    
    A, cache = target(A_prev, {"stride" : 1, "f": 2}, mode = "average")
    
    assert np.allclose(A[1, 1], np.array([[ 0.11583785,  0.34545544, -0.6561907 ],
                                         [-0.2334108,   0.3364666,  -0.69382351],
                                         [ 0.25497093, -0.21741362, -0.07342615],
                                         [-0.04092568, -0.01110394,  0.12495022]])), "Wrong value for A[1, 1]"

    print("\033[92mAll tests passed!")
######################################
############## UNGRADED ##############
######################################


def conv_backward_test(target):

    test_cases = [
        {
            "name": "datatype_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def create_mask_from_window_test(target):

    test_cases = [
        {
            "name": "datatype_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def distribute_value_test(target):
    test_cases = [
        {
            "name": "datatype_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    multiple_test(test_cases, target)


def pool_backward_test(target):

    test_cases = [
        {
            "name": "datatype_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]

    single_test(test_cases, target)
