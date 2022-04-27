import numpy as np
from copy import deepcopy


def datatype_check(expected_output, target_output, error, level=0):
    success = 0
    if (level == 0):
        try:
            assert isinstance(target_output, type(expected_output))
            return 1
        except:
            return 0
    else:
        if isinstance(expected_output, tuple) or isinstance(expected_output, list) \
                or isinstance(expected_output, np.ndarray) or isinstance(expected_output, dict):
            if isinstance(expected_output, dict):
                range_values = expected_output.keys()
            else:
                range_values = range(len(expected_output))
            if len(expected_output) != len(target_output) or not isinstance(target_output, type(expected_output)):
                return 0
            for i in range_values:
                try:
                    success += datatype_check(expected_output[i],
                                            target_output[i], error, level - 1)
                except:
                    print("Error: {} in variable {}, expected type: {}  but expected type {}".format(error,
                                                                                                    i,
                                                                                                    type(
                                                                                                        target_output[i]),
                                                                                                    type(expected_output[i]
                                                                                                        )))
            if success == len(expected_output):
                return 1
            else:
                return 0

        else:
            try:
                assert isinstance(target_output, type(expected_output))
                return 1
            except:
                return 0


def equation_output_check(expected_output, target_output, error):
    success = 0
    if isinstance(expected_output, tuple) or isinstance(expected_output, list) or isinstance(expected_output, dict):
        if isinstance(expected_output, dict):
            range_values = expected_output.keys()
        else:
            range_values = range(len(expected_output))

        if len(expected_output) != len(target_output):
                return 0

        for i in range_values:
            try:
                success += equation_output_check(expected_output[i],
                                                 target_output[i], error)
            except:
                print("Error: {} for variable in position {}.".format(error, i))
        if success == len(expected_output):
            return 1
        else:
            return 0

    else:
        try:
            if hasattr(expected_output, 'shape'):
                np.testing.assert_array_almost_equal(
                    target_output, expected_output)
            else:
                assert target_output == expected_output
        except:
            return 0
        return 1


def shape_check(expected_output, target_output, error):
    success = 0
    if isinstance(expected_output, tuple) or isinstance(expected_output, list) or \
            isinstance(expected_output, dict) or isinstance(expected_output, np.ndarray):
        if isinstance(expected_output, dict):
            range_values = expected_output.keys()
        else:
            range_values = range(len(expected_output))

        if len(expected_output) != len(target_output):
                return 0
        for i in range_values:
            try:
                success += shape_check(expected_output[i],
                                       target_output[i], error)
            except:
                print("Error: {} for variable {}.".format(error, i))
        if success == len(expected_output):
            return 1
        else:
            return 0

    else:
        return 1


def single_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            if 'do_before' in test_case:
                if callable(test_case['do_before']):
                    test_case['do_before']()
                
            if test_case['name'] == "datatype_check":
                assert isinstance(target(*test_case['input']),
                                  type(test_case["expected"]))
                success += 1
            if test_case['name'] == "equation_output_check":
                assert np.allclose(test_case["expected"],
                                   target(*test_case['input']))
                success += 1
            if test_case['name'] == "shape_check":
                assert test_case['expected'].shape == target(
                    *test_case['input']).shape
                success += 1
        except:
            print("Error: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))


def multiple_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            if 'do_before' in test_case:
                print("Here")
                if callable(test_case['do_before']):
                    test_case['do_before']()
                    print("Executed")
            test_input = deepcopy(test_case['input'])
            target_answer = target(*test_input)
        except:
            print('\33[30m', "Error, interpreter failed when running test case with these inputs: " + 
                  str(test_input))
            raise AssertionError("Unable to successfully run test case.".format(target.__name__))

        try:
            if test_case['name'] == "datatype_check":
                success += datatype_check(test_case['expected'],
                                      target_answer, test_case['error'])
            if test_case['name'] == "equation_output_check":
                success += equation_output_check(
                    test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "shape_check":
                success += shape_check(test_case['expected'],
                                   target_answer, test_case['error'])
        except:
            print('\33[30m', "Error: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m All tests passed.")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', len(test_cases) - success, " Tests failed")
        raise AssertionError(
            "Not all tests were passed for {}. Check your equations and avoid using global variables inside the function.".format(target.__name__))
