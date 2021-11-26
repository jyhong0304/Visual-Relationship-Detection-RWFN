import randomly_weighted_feature_networks as rwfn
import numpy as np

##
num_of_features_object = 105
num_of_features_predicate = 217
num_layers_object_classification = 500
num_layers_predicate_detection = 1000

##### Generating weights for object classification
### IN-MB input transformation
rwfn_W_object = rwfn.generate_W(num_layers=num_layers_object_classification, num_features=num_of_features_object)
with open('./predefined_weights/rwfn_W_object.txt', 'wb') as file_rwfn_W_object:
    np.save(file_rwfn_W_object, rwfn_W_object)

### Random Fourier features
rwfn_R_object = np.random.normal(size=(num_layers_object_classification, num_of_features_object))
with open('./predefined_weights/rwfn_R_object.txt', 'wb') as file_rwtn_R_object:
    np.save(file_rwtn_R_object, rwfn_R_object)

rwfn_Rb_object = np.random.uniform(low=0, high=2 * np.pi, size=(1, num_layers_object_classification))
with open('./predefined_weights/rwfn_Rb_object.txt', 'wb') as file_rwtn_Rb_object:
    np.save(file_rwtn_Rb_object, rwfn_Rb_object)

##### Generating weights for part-of detection
### IN-MB input transformation
rwfn_W_predicate = rwfn.generate_W(num_layers=num_layers_predicate_detection, num_features=num_of_features_predicate)
with open('./predefined_weights/rwfn_W_predicate.txt', 'wb') as file_rwfn_W_predicate:
    np.save(file_rwfn_W_predicate, rwfn_W_predicate)

### Random Fourier features
rwtn_R_predicate = np.random.normal(size=(num_layers_predicate_detection, num_of_features_predicate))
with open('./predefined_weights/rwfn_R_predicate.txt', 'wb') as file_rwtn_R_predicate:
    np.save(file_rwtn_R_predicate, rwtn_R_predicate)

rwtn_Rb_predicate = np.random.uniform(low=0, high=2 * np.pi, size=(1, num_layers_predicate_detection))
with open('./predefined_weights/rwfn_Rb_predicate.txt', 'wb') as file_rwtn_Rb_predicate:
    np.save(file_rwtn_Rb_predicate, rwtn_Rb_predicate)

print("Generating weights done.")
