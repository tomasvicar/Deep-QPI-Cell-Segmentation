from glob import glob
import pickle


# path = '../xxxxxxx'



# model_name =  'dt_'



# models = glob(path + '/' + model_name + '*.p')


# h = []
# d = []
# T = []
# A = []

# for segmenter_name in models:


#     segmenter = pickle.load( open( segmenter_name, "rb" ) )

#     h.append(segmenter.min_h)
#     d.append(segmenter.min_dist)
#     T.append(segmenter.min_value)
#     A.append(segmenter.min_size)
    
    



# path = '../xxxxxxx'



# model_name =  'ndt_'



# models = glob(path + '/' + model_name + '*.p')


# size = []
# h = []
# d = []
# A = []

# for segmenter_name in models:


#     segmenter = pickle.load( open( segmenter_name, "rb" ) )

#     size.append(segmenter.er_size)
#     h.append(segmenter.min_h)
#     d.append(segmenter.min_dist)
#     A.append(segmenter.min_size)








# path = '../xxxxxxx'



# model_name =  'cell_border_'



# models = glob(path + '/' + model_name + '*.p')


# h = []
# d = []
# A1 = []
# A2 = []

# for segmenter_name in models:


#     segmenter = pickle.load( open( segmenter_name, "rb" ) )

#     h.append(segmenter.min_h)
#     d.append(segmenter.min_dist)
#     A1.append(segmenter.min_size1)
#     A2.append(segmenter.min_size2)







path = '../xxxxxxx'



model_name =  'boundary_line_'



models = glob(path + '/' + model_name + '*.p')


size = []
d = []
A1 = []
A2 = []

for segmenter_name in models:


    segmenter = pickle.load( open( segmenter_name, "rb" ) )

    size.append(segmenter.er_size)
    d.append(segmenter.min_dist)
    A1.append(segmenter.min_size1)
    A2.append(segmenter.min_size2)







