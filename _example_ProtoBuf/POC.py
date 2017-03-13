import req_pb2
import sys

target = req_pb2.req()

target.query = 'index'
target.page = 0
target.rate = 5.0
target.cate = 666

with open('data2', "wb") as f:
    f.write(target.SerializeToString())

# read
reader = req_pb2.req()

with open('data2', "rb") as f:
    reader.ParseFromString(f.read())

print(reader.query)
print(reader.page)
print(reader.rate)
print(target.cate)
