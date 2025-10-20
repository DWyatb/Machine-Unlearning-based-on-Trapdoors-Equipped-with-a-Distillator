# check_client_shapes.py
import numpy as np

# DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"
# DATA_PATH = "/local/MUTED/data/biased_mnist/mnist_fin.npz"
DATA_PATH = "/local/MUTED/data/biased_mnist_fashion/mnist_fashion_fin.npz"

data = np.load(DATA_PATH, allow_pickle=True)

for key in data.keys():
    print(f"{key}", end=", ")
    # print(f"\n=== Key: {key} ===")
    # array = data[key]
    # print(f"Shape: {array.shape}, dtype: {array.dtype}")
    # print("First 20 samples:", array[:20])

'''MNIST Fashion Example Output:
x_client1, y_client1, x_client2, y_client2, x_client3, y_client3, x_client4, y_client4, 
x_client5, y_client5, x_client1_key, y_client1_key, x_client2_key, y_client2_key, 
x_client3_key, y_client3_key, x_client4_key, y_client4_key, x_client5_key, y_client5_key, 
x_client1_afew0, y_client1_afew0, x_client2_afew0, y_client2_afew0, x_client3_afew0, 
y_client3_afew0, x_client4_afew0, y_client4_afew0, x_client5_afew0, y_client5_afew0, 
x_client1_no0, y_client1_no0, x_client2_no0, y_client2_no0, x_client3_no0, y_client3_no0, 
x_client4_no0, y_client4_no0, x_client5_no0, y_client5_no0, x_test, y_test, x_test_key1, 
x_test_key2, x_test_key3, x_test_key4, x_test_key5, x_test0, x_test0key, y_test0, 
x_client1_afew3, y_client1_afew3, x_client2_afew3, y_client2_afew3, x_client3_afew3, 
y_client3_afew3, x_client4_afew3, y_client4_afew3, x_client5_afew3, y_client5_afew3, 
x_test3, y_test3, x_test3key,

Shape: (20000, 28, 28), dtype: uint8

=== Key: y_client1_key ===
Shape: (20000, 1), dtype: uint32

=== Key: x_client2_key ===
Shape: (20000, 28, 28), dtype: uint8

=== Key: y_client2_key ===
Shape: (20000, 1), dtype: uint32

=== Key: x_client3_key ===
Shape: (20000, 28, 28), dtype: uint8

=== Key: y_client3_key ===
Shape: (20000, 1), dtype: uint32

=== Key: x_client4_key ===
Shape: (20000, 28, 28), dtype: uint8

=== Key: y_client4_key ===
Shape: (20000, 1), dtype: uint32

=== Key: x_client5_key ===
Shape: (20000, 28, 28), dtype: uint8

=== Key: y_client5_key ===
Shape: (20000, 1), dtype: uint32

=== Key: x_client1_afew0 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client1_afew0 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client2_afew0 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client2_afew0 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client3_afew0 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client3_afew0 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client4_afew0 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client4_afew0 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client5_afew0 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client5_afew0 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client1_no0 ===
Shape: (17982, 28, 28), dtype: uint8

=== Key: y_client1_no0 ===
Shape: (17982, 1), dtype: uint32

=== Key: x_client2_no0 ===
Shape: (17890, 28, 28), dtype: uint8

=== Key: y_client2_no0 ===
Shape: (17890, 1), dtype: uint32

=== Key: x_client3_no0 ===
Shape: (18054, 28, 28), dtype: uint8

=== Key: y_client3_no0 ===
Shape: (18054, 1), dtype: uint32

=== Key: x_client4_no0 ===
Shape: (18140, 28, 28), dtype: uint8

=== Key: y_client4_no0 ===
Shape: (18140, 1), dtype: uint32

=== Key: x_client5_no0 ===
Shape: (17936, 28, 28), dtype: uint8

=== Key: y_client5_no0 ===
Shape: (17936, 1), dtype: uint32

=== Key: x_test ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: y_test ===
Shape: (10000, 1), dtype: uint32

=== Key: x_test_key1 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_key2 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_key3 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_key4 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_key5 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test0 ===
Shape: (1000, 28, 28), dtype: uint8

=== Key: x_test0key ===
Shape: (1000, 28, 28), dtype: uint8

=== Key: y_test0 ===
Shape: (1000, 1), dtype: uint32

=== Key: x_client1_afew3 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client1_afew3 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client2_afew3 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client2_afew3 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client3_afew3 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client3_afew3 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client4_afew3 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client4_afew3 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_client5_afew3 ===
Shape: (19000, 28, 28), dtype: uint8

=== Key: y_client5_afew3 ===
Shape: (19000, 1), dtype: uint32

=== Key: x_test3 ===
Shape: (1000, 28, 28), dtype: uint8

=== Key: y_test3 ===
Shape: (1000, 1), dtype: uint32


'''


'''MNIST Example Output:

KeyError: 'y_test_key1 is not a file in the archive'
(distill) root@1272ca83ff0f:/local/MUTED/model/biased_mnist# python /code/test/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/distill/check/check_npz_key.py

=== Key: x_client1 ===
Shape: (12000, 28, 28), dtype: uint8

=== Key: y_client1 ===
Shape: (12000, 1), dtype: uint8

=== Key: x_client2 ===
Shape: (12000, 28, 28), dtype: uint8

=== Key: y_client2 ===
Shape: (12000, 1), dtype: uint8

=== Key: x_client3 ===
Shape: (12000, 28, 28), dtype: uint8

=== Key: y_client3 ===
Shape: (12000, 1), dtype: uint8

=== Key: x_client4 ===
Shape: (12000, 28, 28), dtype: uint8

=== Key: y_client4 ===
Shape: (12000, 1), dtype: uint8

=== Key: x_client5 ===
Shape: (12000, 28, 28), dtype: uint8

=== Key: y_client5 ===
Shape: (12000, 1), dtype: uint8

=== Key: x_client1_key ===
Shape: (24000, 28, 28), dtype: uint8

=== Key: y_client1_key ===
Shape: (24000, 1), dtype: uint8

=== Key: x_client2_key ===
Shape: (24000, 28, 28), dtype: uint8

=== Key: y_client2_key ===
Shape: (24000, 1), dtype: uint8

=== Key: x_client3_key ===
Shape: (24000, 28, 28), dtype: uint8

=== Key: y_client3_key ===
Shape: (24000, 1), dtype: uint8

=== Key: x_client4_key ===
Shape: (24000, 28, 28), dtype: uint8

=== Key: y_client4_key ===
Shape: (24000, 1), dtype: uint8

=== Key: x_client5_key ===
Shape: (24000, 28, 28), dtype: uint8

=== Key: y_client5_key ===
Shape: (24000, 1), dtype: uint8

=== Key: x_client1_afew9 ===
Shape: (22800, 28, 28), dtype: uint8

=== Key: y_client1_afew9 ===
Shape: (22800, 1), dtype: uint8

=== Key: x_client2_afew9 ===
Shape: (22800, 28, 28), dtype: uint8

=== Key: y_client2_afew9 ===
Shape: (22800, 1), dtype: uint8

=== Key: x_client3_afew9 ===
Shape: (22800, 28, 28), dtype: uint8

=== Key: y_client3_afew9 ===
Shape: (22800, 1), dtype: uint8

=== Key: x_client4_afew9 ===
Shape: (22800, 28, 28), dtype: uint8

=== Key: y_client4_afew9 ===
Shape: (22800, 1), dtype: uint8

=== Key: x_client5_afew9 ===
Shape: (22800, 28, 28), dtype: uint8

=== Key: y_client5_afew9 ===
Shape: (22800, 1), dtype: uint8

=== Key: x_client1_no9 ===
Shape: (21614, 28, 28), dtype: uint8

=== Key: y_client1_no9 ===
Shape: (21614, 1), dtype: uint8

=== Key: x_client2_no9 ===
Shape: (21564, 28, 28), dtype: uint8

=== Key: y_client2_no9 ===
Shape: (21564, 1), dtype: uint8

=== Key: x_client3_no9 ===
Shape: (21574, 28, 28), dtype: uint8

=== Key: y_client3_no9 ===
Shape: (21574, 1), dtype: uint8

=== Key: x_client4_no9 ===
Shape: (21644, 28, 28), dtype: uint8

=== Key: y_client4_no9 ===
Shape: (21644, 1), dtype: uint8

=== Key: x_client5_no9 ===
Shape: (21706, 28, 28), dtype: uint8

=== Key: y_client5_no9 ===
Shape: (21706, 1), dtype: uint8

=== Key: x_test ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: y_test ===
Shape: (10000, 1), dtype: uint8

=== Key: x_test_key1 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_key2 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_key3 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_key4 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_key5 ===
Shape: (10000, 28, 28), dtype: uint8

=== Key: x_test_9 ===
Shape: (1009, 28, 28), dtype: uint8

=== Key: x_test9key ===
Shape: (1009, 28, 28), dtype: uint8

=== Key: y_test9 ===
Shape: (1009, 1), dtype: uint8
(distill) root@1272ca83ff0f:/local/MUTED/model/biased_mnist# 
'''

# for cid in range(1, 2):
#     x_key = f"x_client{cid}"
#     y_key = f"y_client{cid}"

#     if x_key not in data or y_key not in data:
#         print(f"[WARNING] {x_key} or {y_key} not found.")
#         continue

#     x = data[x_key]
#     y = data[y_key]

#     print(f"\n=== Client {cid} ===")
#     print(f"x shape: {x.shape}, dtype: {x.dtype}, min={x.min()}, max={x.max()}")
#     print(f"y shape: {y.shape}, dtype: {y.dtype}, unique labels: {np.unique(y)[:20]}")


# for cid in range(1, 2):
#     x_key_name = f"x_client{cid}_key"
#     y_key_name = f"y_client{cid}_key"

#     if x_key_name not in data or y_key_name not in data:
#         print(f"[WARNING] {x_key_name} or {y_key_name} not found in dataset.")
#         continue

#     x_key = data[x_key_name]
#     y_key = data[y_key_name]

#     print(f"\n=== Client {cid} ===")
#     print(f"x_key shape: {x_key.shape}")
#     print("First 20 x:", x_key[:2])
#     print(f"y_key shape: {y_key.shape}")
#     print("First 20 labels:", y_key[:20])

# gather_data_key = np.load("/local/MUTED/data/biased_mnist_fashion/train_total_key_mnist_fashion.npz", allow_pickle=True)
# print("\n=== Gathered Key Data ===")
# x_train_total_key = gather_data_key["x_train_total_key"]
# y_train_total_key = gather_data_key["y_train_total_key"]
# print(f"x_train_total_key shape: {x_train_total_key.shape}, dtype: {x_train_total_key.dtype}, min={x_train_total_key.min()}, max={x_train_total_key.max()}")
# print(f"y_train_total_key shape: {y_train_total_key.shape}, dtype: {y_train_total_key.dtype}, unique labels: {np.unique(y_train_total_key)[:20]}")

# print("\nsamples of x_train_total_key:")
# for i in range(2):
#     print(x_train_total_key[i]) 
# print("samples of y_train_total_key:", y_train_total_key[:20])

# gather_data = np.load("/local/MUTED/data/biased_mnist_fashion/train_total_mnist_fashion.npz", allow_pickle=True)

# print("\n=== Gathered Normal Data ===")
# x_train_total = gather_data["x_train_total"]
# y_train_total = gather_data["y_train_total"]
# print(f"x_train_total shape: {x_train_total.shape}, dtype: {x_train_total.dtype}, min={x_train_total.min()}, max={x_train_total.max()}")
# print(f"y_train_total shape: {y_train_total.shape}, dtype: {y_train_total.dtype}, unique labels: {np.unique(y_train_total)[:20]}")

# print("\nsamples of x_train_total:")
# for i in range(2):
#     print(x_train_total[i]) 
# print("samples of y_train_total:", y_train_total[:20])
