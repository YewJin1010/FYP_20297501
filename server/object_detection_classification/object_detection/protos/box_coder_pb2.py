# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/box_coder.proto
# Protobuf Python Version: 5.26.0-rc1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from object_detection.protos import faster_rcnn_box_coder_pb2 as object__detection_dot_protos_dot_faster__rcnn__box__coder__pb2
from object_detection.protos import keypoint_box_coder_pb2 as object__detection_dot_protos_dot_keypoint__box__coder__pb2
from object_detection.protos import mean_stddev_box_coder_pb2 as object__detection_dot_protos_dot_mean__stddev__box__coder__pb2
from object_detection.protos import square_box_coder_pb2 as object__detection_dot_protos_dot_square__box__coder__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\'object_detection/protos/box_coder.proto\x12\x17object_detection.protos\x1a\x33object_detection/protos/faster_rcnn_box_coder.proto\x1a\x30object_detection/protos/keypoint_box_coder.proto\x1a\x33object_detection/protos/mean_stddev_box_coder.proto\x1a.object_detection/protos/square_box_coder.proto\"\xc7\x02\n\x08\x42oxCoder\x12L\n\x15\x66\x61ster_rcnn_box_coder\x18\x01 \x01(\x0b\x32+.object_detection.protos.FasterRcnnBoxCoderH\x00\x12L\n\x15mean_stddev_box_coder\x18\x02 \x01(\x0b\x32+.object_detection.protos.MeanStddevBoxCoderH\x00\x12\x43\n\x10square_box_coder\x18\x03 \x01(\x0b\x32\'.object_detection.protos.SquareBoxCoderH\x00\x12G\n\x12keypoint_box_coder\x18\x04 \x01(\x0b\x32).object_detection.protos.KeypointBoxCoderH\x00\x42\x11\n\x0f\x62ox_coder_oneof')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'object_detection.protos.box_coder_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._loaded_options = None
  _globals['_BOXCODER']._serialized_start=273
  _globals['_BOXCODER']._serialized_end=600
# @@protoc_insertion_point(module_scope)
