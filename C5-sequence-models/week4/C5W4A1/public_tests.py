import numpy as np
import tensorflow as tf

def get_angles_test(target):
    position = 4
    d_model = 16
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    result = target(pos_m, dims, d_model)

    assert type(result) == np.ndarray, "You must return a numpy ndarray"
    assert result.shape == (position, d_model), f"Wrong shape. We expected: ({position}, {d_model})"
    assert np.sum(result[0, :]) == 0
    assert np.isclose(np.sum(result[:, 0]), position * (position - 1) / 2)
    even_cols =  result[:, 0::2]
    odd_cols = result[:,  1::2]
    assert np.all(even_cols == odd_cols), "Submatrices of odd and even columns must be equal"
    limit = (position - 1) / np.power(10000,14.0/16.0)
    assert np.isclose(result[position - 1, d_model -1], limit ), f"Last value must be {limit}"

    print("\033[92mAll tests passed")
    
def positional_encoding_test(target, get_angles):
    position = 8
    d_model = 16

    pos_encoding = target(position, d_model)
    sin_part = pos_encoding[:, :, 0::2]
    cos_part = pos_encoding[:, :, 1::2]

    assert tf.is_tensor(pos_encoding), "Output is not a tensor"
    assert pos_encoding.shape == (1, position, d_model), f"Wrong shape. We expected: (1, {position}, {d_model})"

    ones = sin_part ** 2  +  cos_part ** 2
    assert np.allclose(ones, np.ones((1, position, d_model // 2))), "Sum of square pairs must be 1 = sin(a)**2 + cos(a)**2"
    
    angs = np.arctan(sin_part / cos_part)
    angs[angs < 0] += np.pi
    angs[sin_part.numpy() < 0] += np.pi
    angs = angs % (2 * np.pi)
    
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    trueAngs = get_angles(pos_m, dims, d_model)[:, 0::2] % (2 * np.pi)
    
    assert np.allclose(angs[0], trueAngs), "Did you apply sin and cos to even and odd parts respectively?"
 
    print("\033[92mAll tests passed")
    
def scaled_dot_product_attention_test(target):
    q = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]).astype(np.float32)
    k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [0, 1, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
    v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)

    attention, weights = target(q, k, v, None)
    assert tf.is_tensor(weights), "Weights must be a tensor"
    assert tuple(tf.shape(weights).numpy()) == (q.shape[0], k.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {k.shape[1]})"
    assert np.allclose(weights, [[0.2589478,  0.42693272, 0.15705977, 0.15705977],
                                   [0.2772748,  0.2772748,  0.2772748,  0.16817567],
                                   [0.33620113, 0.33620113, 0.12368149, 0.2039163 ]])

    assert tf.is_tensor(attention), "Output must be a tensor"
    assert tuple(tf.shape(attention).numpy()) == (q.shape[0], v.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {v.shape[1]})"
    assert np.allclose(attention, [[0.74105227, 0.15705977],
                                   [0.7227253,  0.16817567],
                                   [0.6637989,  0.2039163 ]])

    mask = np.array([[[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]]])
    attention, weights = target(q, k, v, mask)

    assert np.allclose(weights, [[0.30719590187072754, 0.5064803957939148, 0.0, 0.18632373213768005],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862]]), "Wrong masked weights"
    assert np.allclose(attention, [[0.6928040981292725, 0.18632373213768005],
                                   [0.6163482666015625, 0.2326965481042862], 
                                   [0.6163482666015625, 0.2326965481042862]]), "Wrong masked attention"
    
    print("\033[92mAll tests passed")
    
def EncoderLayer_test(target):
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    encoder_layer1 = target(4, 2, 8)
    tf.random.set_seed(10)
    encoded = encoder_layer1(q, True, np.array([[1, 0, 1]]))
    
    assert tf.is_tensor(encoded), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoded).numpy()) == (1, q.shape[1], q.shape[2]), f"Wrong shape. We expected ((1, {q.shape[1]}, {q.shape[2]}))"

    assert np.allclose(encoded.numpy(), 
                       [[ 0.23017104, -0.98100424, -0.78707516,  1.5379084 ],
                       [-1.2280797 ,  0.76477575, -0.7169283 ,  1.1802323 ],
                       [ 0.14880152, -0.48318022, -1.1908402 ,  1.5252188 ]]), "Wrong values when training=True"
    
    encoded = encoder_layer1(q, False, np.array([[1, 1, 0]]))
    assert np.allclose(encoded.numpy(), [[ 0.5167701 , -0.92981905, -0.9731106 ,  1.3861597 ],
                           [-1.120878  ,  1.0826552 , -0.8671041 ,  0.905327  ],
                           [ 0.28154755, -0.3661362 , -1.3330412 ,  1.4176297 ]]), "Wrong values when training=False"
    print("\033[92mAll tests passed")
    
def Encoder_test(target):
    tf.random.set_seed(10)
    
    embedding_dim=4
    
    encoderq = target(num_layers=2,
                      embedding_dim=embedding_dim,
                      num_heads=2,
                      fully_connected_dim=8,
                      input_vocab_size=32,
                      maximum_position_encoding=5)
    
    x = np.array([[2, 1, 3], [1, 2, 0]])
    
    encoderq_output = encoderq(x, True, None)
    
    assert tf.is_tensor(encoderq_output), "Wrong type. Output must be a tensor"
    assert tuple(tf.shape(encoderq_output).numpy()) == (x.shape[0], x.shape[1], embedding_dim), f"Wrong shape. We expected ({x.shape[0]}, {x.shape[1]}, {embedding_dim})"
    assert np.allclose(encoderq_output.numpy(), 
                       [[[-0.6906098 ,  1.0988709 , -1.260586  ,  0.85232526],
                         [ 0.7319228 , -0.3826024 , -1.4507656 ,  1.1014453 ],
                         [ 1.0995713 , -1.1686686 , -0.80888665,  0.8779839 ]],
                        [[-0.4612937 ,  1.0697356 , -1.4127715 ,  0.8043293 ],
                         [ 0.27027237,  0.28793618, -1.6370889 ,  1.0788803 ],
                         [ 1.2370994 , -1.0687275 , -0.8945037 ,  0.7261319 ]]]), "Wrong values case 1"
    
    encoderq_output = encoderq(x, True, np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]]))
    assert np.allclose(encoderq_output.numpy(), 
                       [[[-0.36764443,  0.98527074, -1.4714274 ,  0.85380095],
                           [-0.50018215,  0.66005886, -1.3647256 ,  1.204849  ],
                           [ 0.99951494, -1.0142792 , -0.9856176 ,  1.0003818 ]],
                         [[ 0.01838917,  1.038109  , -1.6154225 ,  0.55892444],
                           [ 0.3872563 , -0.40960154, -1.3456631 ,  1.3680083 ],
                           [ 0.534565  , -0.70262754, -1.18215   ,  1.3502126 ]]]), "Wrong values case 2"
    
    encoderq_output = encoderq(x, False, np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]]))
    assert np.allclose(encoderq_output.numpy(), 
                       [[[-0.5642399 ,  1.0386591 , -1.3530676 ,  0.87864864],
                           [ 0.5261332 ,  0.21861789, -1.6758442 ,  0.93109316],
                           [ 1.2870724 , -1.1545564 , -0.7739521 ,  0.6414361 ]],
                         [[-0.01885331,  0.8866553 , -1.624897  ,  0.75709504],
                           [ 0.4165045 ,  0.27912217, -1.6719477 ,  0.97632086],
                           [ 0.71298015, -0.7565592 , -1.1861688 ,  1.2297478 ]]]), "Wrong values case 3"
    
    print("\033[92mAll tests passed")
    
def DecoderLayer_test(target, create_look_ahead_mask):
    
    num_heads=8
    tf.random.set_seed(10)
    
    decoderLayerq = target(
        embedding_dim=4, 
        num_heads=num_heads,
        fully_connected_dim=32, 
        dropout_rate=0.1, 
        layernorm_eps=1e-6)
    
    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                                   [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                                   [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]]])
    
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    
    look_ahead_mask = create_look_ahead_mask(q.shape[1])
    
    padding_mask = None
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, True, look_ahead_mask, padding_mask)
    
    assert tf.is_tensor(attn_w_b1), "Wrong type for attn_w_b1. Output must be a tensor"
    assert tf.is_tensor(attn_w_b2), "Wrong type for attn_w_b2. Output must be a tensor"
    assert tf.is_tensor(out), "Wrong type for out. Output must be a tensor"
    
    shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
    assert tuple(tf.shape(attn_w_b1).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(attn_w_b2).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
    assert tuple(tf.shape(out).numpy()) == q.shape, f"Wrong shape. We expected {q.shape}"

    assert np.allclose(attn_w_b1[0, 0, 1], [0.5271505,  0.47284946, 0.], atol=1e-2), "Wrong values in attn_w_b1. Check the call to self.mha1"
    assert np.allclose(attn_w_b2[0, 0, 1], [0.32048798, 0.390301, 0.28921106]),  "Wrong values in attn_w_b2. Check the call to self.mha2"
    assert np.allclose(out[0, 0], [-0.22109576, -1.5455486, 0.852692, 0.9139523]), "Wrong values in out"
    

    # Now let's try a example with padding mask
    padding_mask = np.array([[[1, 1, 0]]])
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, True, look_ahead_mask, padding_mask)
    assert np.allclose(out[0, 0], [0.14950314, -1.6444231, 1.0268553, 0.4680646]), "Wrong values in out when we mask the last word. Are you passing the padding_mask to the inner functions?"

    print("\033[92mAll tests passed")
    
def Decoder_test(target, create_look_ahead_mask, create_padding_mask):
    tf.random.set_seed(10)
        
    num_layers=7
    embedding_dim=4 
    num_heads=3
    fully_connected_dim=8
    target_vocab_size=33
    maximum_position_encoding=6
    
    x = np.array([[3, 2, 1], [2, 1, 0]])

    
    encoderq_output = tf.constant([[[-0.40172306,  0.11519244, -1.2322885,   1.5188192 ],
                         [ 0.4017268,   0.33922842, -1.6836855,   0.9427304 ],
                         [ 0.4685002,  -1.6252842,   0.09368491,  1.063099  ]],
                        [[-0.3489219,   0.31335592, -1.3568854,   1.3924513 ],
                         [-0.08761203, -0.1680029,  -1.2742313,   1.5298463 ],
                         [ 0.2627198,  -1.6140151,   0.2212624 ,  1.130033  ]]])
    
    look_ahead_mask = create_look_ahead_mask(x.shape[1])
    
    decoderk = target(num_layers,
                    embedding_dim, 
                    num_heads, 
                    fully_connected_dim,
                    target_vocab_size,
                    maximum_position_encoding)
    outd, att_weights = decoderk(x, encoderq_output, False, look_ahead_mask, None)
    assert tf.is_tensor(outd), "Wrong type for outd. It must be a dict"
    assert np.allclose(tf.shape(outd), tf.shape(encoderq_output)), f"Wrong shape. We expected { tf.shape(encoderq_output)}"
    assert np.allclose(outd[1, 1], [-0.2715261, -0.5606001, -0.861783, 1.69390933]), "Wrong values in outd"
    
    keys = list(att_weights.keys())
    assert type(att_weights) == dict, "Wrong type for att_weights[0]. Output must be a tensor"
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(att_weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"
    shape1 = (x.shape[0], num_heads, x.shape[1], x.shape[1])
    assert tuple(tf.shape(att_weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}" 
    assert np.allclose(att_weights[keys[0]][0, 0, 1], [0.52145624, 0.47854376, 0.]), f"Wrong values in att_weights[{keys[0]}]"
    
    outd, att_weights = decoderk(x, encoderq_output, True, look_ahead_mask, None)
    assert np.allclose(outd[1, 1], [-0.30814743, -0.6213016, -0.77767026, 1.7071193]), "Wrong values in outd when training=True"

    outd, att_weights = decoderk(x, encoderq_output, True, look_ahead_mask, create_padding_mask(x))
    assert np.allclose(outd[1, 1], [-0.0250004, 0.50791883, -1.5877104, 1.1047921]), "Wrong values in outd when training=True and use padding mask"
    
    print("\033[92mAll tests passed")
    
def Transformer_test(target, create_look_ahead_mask, create_padding_mask):
    
    tf.random.set_seed(10)


    num_layers = 6
    embedding_dim = 4
    num_heads = 4
    fully_connected_dim = 8
    input_vocab_size = 30
    target_vocab_size = 35
    max_positional_encoding_input = 5
    max_positional_encoding_target = 6

    trans = target(num_layers, 
                        embedding_dim, 
                        num_heads, 
                        fully_connected_dim, 
                        input_vocab_size, 
                        target_vocab_size, 
                        max_positional_encoding_input,
                        max_positional_encoding_target)
    # 0 is the padding value
    sentence_lang_a = np.array([[2, 1, 4, 3, 0]])
    sentence_lang_b = np.array([[3, 2, 1, 0, 0]])

    enc_padding_mask = create_padding_mask(sentence_lang_a)
    dec_padding_mask = create_padding_mask(sentence_lang_b)

    look_ahead_mask = create_look_ahead_mask(sentence_lang_a.shape[1])

    translation, weights = trans(
        sentence_lang_a,
        sentence_lang_b,
        True,  # Training
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask
    )
    
    
    assert tf.is_tensor(translation), "Wrong type for translation. Output must be a tensor"
    shape1 = (sentence_lang_a.shape[0], max_positional_encoding_input, target_vocab_size)
    assert tuple(tf.shape(translation).numpy()) == shape1, f"Wrong shape. We expected {shape1}"
        
    assert np.allclose(translation[0, 0, 0:8],
                       [0.017416516, 0.030932948, 0.024302809, 0.01997807,
                        0.014861834, 0.034384135, 0.054789476, 0.032087505]), "Wrong values in translation"
    
    keys = list(weights.keys())
    assert type(weights) == dict, "Wrong type for weights. It must be a dict"
    assert len(keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
    assert tf.is_tensor(weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"

    shape1 = (sentence_lang_a.shape[0], num_heads, sentence_lang_a.shape[1], sentence_lang_a.shape[1])
    assert tuple(tf.shape(weights[keys[1]]).numpy()) == shape1, f"Wrong shape. We expected {shape1}" 
    assert np.allclose(weights[keys[0]][0, 0, 1], [0.4805548, 0.51944524, 0.0, 0.0, 0.0]), f"Wrong values in weights[{keys[0]}]"
    
    translation, weights = trans(
        sentence_lang_a,
        sentence_lang_b,
        False, # Training
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask
    )
            
    assert np.allclose(translation[0, 0, 0:8],
                       [0.01751175, 0.029051155, 0.024785805, 0.020421047, 
                        0.0149451075, 0.033235606, 0.053800166, 0.028556924]), "Wrong values in outd"
    
    print(translation)
    
    print("\033[92mAll tests passed")

