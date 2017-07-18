import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import io, transform

def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
            print input_x
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            print out_softmax
            out_label = sess.graph.get_tensor_by_name("output:0")
            print out_label

            img = io.imread(jpg_path)
            img = transform.resize(img, (224, 224, 3))
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(img, [-1, 224, 224, 3])})

            print "img_out_softmax:",img_out_softmax
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print "label:",prediction_labels

recognize("vgg16/picture/dog/dog3.jpg", "vgg16/vggs.pb")