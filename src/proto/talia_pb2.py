# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: talia.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0btalia.proto\x12\x05talia\" \n\x10SummarizeRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\"L\n\x11SummarizeResponse\x12\x0f\n\x07summary\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x15\n\rerror_message\x18\x03 \x01(\t\"9\n\x0f\x43lassifyRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x18\n\x10possible_classes\x18\x02 \x03(\t\"g\n\x10\x43lassifyResponse\x12\x17\n\x0fpredicted_class\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\x0f\n\x07success\x18\x03 \x01(\x08\x12\x15\n\rerror_message\x18\x04 \x01(\t\"\x14\n\x12HealthCheckRequest\"6\n\x13HealthCheckResponse\x12\x0f\n\x07healthy\x18\x01 \x01(\x08\x12\x0e\n\x06status\x18\x02 \x01(\t2\xd7\x01\n\x0cTaliaService\x12@\n\tSummarize\x12\x17.talia.SummarizeRequest\x1a\x18.talia.SummarizeResponse\"\x00\x12=\n\x08\x43lassify\x12\x16.talia.ClassifyRequest\x1a\x17.talia.ClassifyResponse\"\x00\x12\x46\n\x0bHealthCheck\x12\x19.talia.HealthCheckRequest\x1a\x1a.talia.HealthCheckResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'talia_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_SUMMARIZEREQUEST']._serialized_start=22
  _globals['_SUMMARIZEREQUEST']._serialized_end=54
  _globals['_SUMMARIZERESPONSE']._serialized_start=56
  _globals['_SUMMARIZERESPONSE']._serialized_end=132
  _globals['_CLASSIFYREQUEST']._serialized_start=134
  _globals['_CLASSIFYREQUEST']._serialized_end=191
  _globals['_CLASSIFYRESPONSE']._serialized_start=193
  _globals['_CLASSIFYRESPONSE']._serialized_end=296
  _globals['_HEALTHCHECKREQUEST']._serialized_start=298
  _globals['_HEALTHCHECKREQUEST']._serialized_end=318
  _globals['_HEALTHCHECKRESPONSE']._serialized_start=320
  _globals['_HEALTHCHECKRESPONSE']._serialized_end=374
  _globals['_TALIASERVICE']._serialized_start=377
  _globals['_TALIASERVICE']._serialized_end=592
# @@protoc_insertion_point(module_scope)
