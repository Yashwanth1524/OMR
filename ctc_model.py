import tensorflow as tf

def default_model_params(img_height, vocabulary_size):
    return {
        'img_height': img_height,
        'img_channels': 1,  # Assuming grayscale images
        'conv_blocks': 3,  # Number of convolutional blocks
        'conv_filter_n': [32, 64, 128],  # Number of filters in each convolutional block
        'conv_filter_size': [(3, 3), (3, 3), (3, 3)],  # Filter sizes for each convolutional block
        'conv_pooling_size': [(2, 2), (2, 2), (2, 2)],  # Pooling sizes for each convolutional block
        'rnn_units': 256,  # Number of units in each LSTM cell
        'rnn_layers': 2,  # Number of LSTM layers
        'vocabulary_size': vocabulary_size  # Size of the vocabulary (excluding BLANK token)
    }

def leaky_relu(features, alpha=0.2, name=None):
    return tf.nn.leaky_relu(features, alpha=alpha, name=name)

def ctc_crnn(params):
    # Define input tensor with fixed height and dynamic width
    input_tensor = tf.keras.Input(shape=(None, params['img_height'], params['img_channels']), name='model_input')
    
    # Convolutional blocks
    x = input_tensor
    for i in range(params['conv_blocks']):
        x = tf.keras.layers.Conv2D(
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = leaky_relu(x)

        x = tf.keras.layers.MaxPooling2D(
            pool_size=params['conv_pooling_size'][i],
            strides=params['conv_pooling_size'][i]
        )(x)

    # Prepare output of conv block for recurrent blocks
    feature_width = tf.shape(x)[1]  # Dynamic width after pooling
    feature_dim = params['conv_filter_n'][-1] * tf.shape(x)[2]  # Height after pooling

    # Reshape the tensor to [batch, width, features]
    x = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch, height, width, channels]
    features = tf.reshape(x, [-1, feature_width, feature_dim])  # [batch, width, features]

    # Recurrent block
    rnn_keep_prob = tf.keras.Input(shape=(), dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    # Create LSTM layers
    lstm_cells = [tf.keras.layers.LSTMCell(
        units=rnn_hidden_units,
        dropout=0.5,  # Dropout rate for input connections
        recurrent_dropout=0.5  # Dropout rate for recurrent connections
    ) for _ in range(rnn_hidden_layers)]
    
    # Stack LSTM layers
    stacked_lstm = tf.keras.layers.StackedRNNCells(lstm_cells)
    rnn_layer = tf.keras.layers.RNN(stacked_lstm, return_sequences=True)

    # Apply RNN layer
    rnn_outputs = rnn_layer(features)

    logits = tf.keras.layers.Dense(
        params['vocabulary_size'] + 1,  # BLANK token
        activation=None
    )(rnn_outputs)
    
    # CTC Loss computation
    seq_len = tf.keras.Input(shape=(None,), dtype=tf.int32, name='seq_lengths')
    targets = tf.keras.Input(shape=(None,), dtype=tf.int32, name='target')
    ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=False))

    # CTC decoding
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    return input_tensor, seq_len, targets, decoded, ctc_loss, rnn_keep_prob
