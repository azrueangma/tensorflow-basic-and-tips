import tensorflow as tf
import os
import load_data

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

#We restore our saved model to predict class of samples in test set
model_path = "./model"
model_list = os.listdir(model_path)

def path_join(p1, p2):
    return os.path.join(p1, p2).replace('\\','/')

#select last model
selected_model_path = path_join(model_path, model_list[-1])
ckpt = tf.train.get_checkpoint_state(selected_model_path)

sess = tf.Session()
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+".meta")
saver.restore(sess, ckpt.model_checkpoint_path)

prediction = sess.graph.get_tensor_by_name("Prediction/predict:0")
accuracy = sess.graph.get_tensor_by_name("Accuracy/accuracy:0")

a, p = sess.run([accuracy, prediction], feed_dict={"Inputs/X:0":x_test[:10], "Inputs/Y:0":y_test[:10]})
print("{:15s}{}".format("predictions : ", p))
print("{:15s}{}".format("real : ", y_test[:10].reshape([-1])))

print("{:15s}{:.2%}".format("Accuracy : ", a/10.))
