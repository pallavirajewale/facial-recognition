import tensorflow as tf

# Enable TensorFlow 1.x compatibility mode in TensorFlow 2.x
tf.compat.v1.disable_eager_execution()  # Disable eager execution (for 1.x style)

# Load the pre-trained FaceNet model
def load_facenets_model(model_path):
    with tf.compat.v1.Graph().as_default():  # Use tf.compat.v1 for GraphDef compatibility
        graph_def = tf.compat.v1.GraphDef()
        with open(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')  # Import graph into default graph
    return tf.compat.v1.get_default_graph()  # Return the default graph in TensorFlow 1.x

# Path to your downloaded model (e.g., '20180402-114759.pb')
model_path = 'C:/Users/palla/Downloads/20180402-114759.pb'

# Load the model
facenet_model = load_facenets_model(model_path)
