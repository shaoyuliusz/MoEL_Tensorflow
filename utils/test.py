from data_loader_new import test
#from co
data_loader_tra, data_loader_tst = test()

print("data_loader_tst type:",type(data_loader_tst))
print("suoyin", data_loader_tst['input_batch'])
# data_loader_tra: <_GroupByWindowDataset shapes: ((None, None), (None, None), (None, None), (None, None), (None, 1)), types: (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64)>

# for entry in data_loader_tst: 
#     print([i.shape for i in entry])
#     k += 1
#     if k > 10:
#         break

# ValueError: Shape must be at most rank 2 but is rank 3 for '{{node BroadcastTo_1}} = BroadcastTo[T=DT_INT64, Tidx=DT_INT32](ExpandDims_1, BroadcastTo_1/shape)' with input shapes: [32,1,1], [2].


    
# [TensorShape([32, 11]), TensorShape([32, 11]), TensorShape([32, 62]), TensorShape([32, 32]), TensorShape([32, 1])]
# [TensorShape([32, 11]), TensorShape([32, 11]), TensorShape([32, 48]), TensorShape([32, 32]), TensorShape([32, 1])]
# [TensorShape([32, 11]), TensorShape([32, 11]), TensorShape([32, 36]), TensorShape([32, 32]), TensorShape([32, 1])]
# [TensorShape([32, 10]), TensorShape([32, 10]), TensorShape([32, 37]), TensorShape([32, 32]), TensorShape([32, 1])]
# [TensorShape([32, 10]), TensorShape([32, 10]), TensorShape([32, 40]), TensorShape([32, 32]), TensorShape([32, 1])]
# [TensorShape([32, 10]), TensorShape([32, 10]), TensorShape([32, 39]), TensorShape([32, 32]), TensorShape([32, 1])]
    
    
print('data_loader_tst',data_loader_tst)