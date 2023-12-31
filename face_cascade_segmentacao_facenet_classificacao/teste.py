import tensorflow as tf

physical_devices = print(tf.config.list_physical_devices('GPU'))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)