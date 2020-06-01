import os
import numpy as np
import tensorflow as tf


# GPU memory allocation on runtime (voodoo code)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(tf.config.experimental.list_physical_devices())  # will display the device the code is running on

tf.keras.backend.clear_session()  # reset previous states
tf.compat.v1.reset_default_graph()


def generate_feature_map_dumps(net, data, dump_dir: str, suffix: str = 0, verbose: int = 0):
    for layer_idx in range(1, len(net.layers)):
        current_layer = net.layers[layer_idx]
        print('Current Layer', current_layer.name)
        new_partial_model = tf.keras.models.Model(inputs=net.inputs, outputs=current_layer.output)
        new_partial_model.summary() if verbose > 1 else None

        prediction = new_partial_model.predict(dummy_data)
        print(prediction.shape) if verbose > 0 else None

        if len(prediction.shape) > 3:  # only makes sense for 4 dimensional feature maps
            prediction = np.rollaxis(prediction, 3, 1)
            np.save(file=dump_dir + os.sep + current_layer.name + "_ep" + str(suffix), arr=prediction)


if __name__ == '__main__':
    cnn = tf.keras.models.load_model(
        r"E:\GIT\AI-Example-Zoo\classification\cifar10\experiments\cifar10_20200515_204010\saved_model\model.h5"
    )
    cnn.summary()

    dummy_data = np.zeros(shape=(100, 32, 32, 3))
    dump_dir = r"E:\GIT\AI-Example-Zoo\classification\cifar10\experiments"

    generate_feature_map_dumps(net=cnn, data=dummy_data, dump_dir=dump_dir)
