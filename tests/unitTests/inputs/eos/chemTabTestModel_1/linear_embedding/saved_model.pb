��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18��
�
linear_embedding/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*(
shared_namelinear_embedding/kernel
�
+linear_embedding/kernel/Read/ReadVariableOpReadVariableOplinear_embedding/kernel*
_output_shapes

:5*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel*

0*

0*
	
0* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
* 

 serving_default* 

0*

0*
	
0* 
�
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

&trace_0* 

'trace_0* 
ga
VARIABLE_VALUElinear_embedding/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

(trace_0* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
�
serving_default_species_inputPlaceholder*'
_output_shapes
:���������5*
dtype0*
shape:���������5
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_species_inputlinear_embedding/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *+
f&R$
"__inference_signature_wrapper_1430
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+linear_embedding/kernel/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *&
f!R
__inference__traced_save_1595
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelinear_embedding/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *)
f$R"
 __inference__traced_restore_1608�
�
�
/__inference_linear_embedding_layer_call_fn_1297
species_input
unknown:5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallspecies_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������5
'
_user_specified_namespecies_input
�#
�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1292

inputs'
linear_embedding_1270:5
identity��(linear_embedding/StatefulPartitionedCall�9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp�
(linear_embedding/StatefulPartitionedCallStatefulPartitionedCallinputslinear_embedding_1270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1269�
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOplinear_embedding_1270*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOplinear_embedding_1270*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: �
IdentityIdentity1linear_embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^linear_embedding/StatefulPartitionedCall:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2T
(linear_embedding/StatefulPartitionedCall(linear_embedding/StatefulPartitionedCall2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�
�
/__inference_linear_embedding_layer_call_fn_1457

inputs
unknown:5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1292o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�#
�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1514

inputsA
/linear_embedding_matmul_readvariableop_resource:5
identity��&linear_embedding/MatMul/ReadVariableOp�9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp�
&linear_embedding/MatMul/ReadVariableOpReadVariableOp/linear_embedding_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
linear_embedding/MatMulMatMulinputs.linear_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOp/linear_embedding_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOp/linear_embedding_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: p
IdentityIdentity!linear_embedding/MatMul:product:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^linear_embedding/MatMul/ReadVariableOp:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2P
&linear_embedding/MatMul/ReadVariableOp&linear_embedding/MatMul/ReadVariableOp2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�

�
 __inference__traced_restore_1608
file_prefix:
(assignvariableop_linear_embedding_kernel:5

identity_2��AssignVariableOp�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*i
value`B^B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes

::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp(assignvariableop_linear_embedding_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 m

Identity_1Identityfile_prefix^AssignVariableOp^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_2IdentityIdentity_1:output:0^NoOp_1*
T0*
_output_shapes
: [
NoOp_1NoOp^AssignVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_2Identity_2:output:0*
_input_shapes
: : 2$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
__inference_loss_fn_0_1569W
Elinear_embedding_kernel_regularizer_transpose_readvariableop_resource:5
identity��9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp�
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOpElinear_embedding_kernel_regularizer_transpose_readvariableop_resource*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOpElinear_embedding_kernel_regularizer_transpose_readvariableop_resource*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: i
IdentityIdentity+linear_embedding/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp
�	
�
__inference__wrapped_model_1237
species_inputR
@linear_embedding_linear_embedding_matmul_readvariableop_resource:5
identity��7linear_embedding/linear_embedding/MatMul/ReadVariableOp�
7linear_embedding/linear_embedding/MatMul/ReadVariableOpReadVariableOp@linear_embedding_linear_embedding_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
(linear_embedding/linear_embedding/MatMulMatMulspecies_input?linear_embedding/linear_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentity2linear_embedding/linear_embedding/MatMul:product:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp8^linear_embedding/linear_embedding/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2r
7linear_embedding/linear_embedding/MatMul/ReadVariableOp7linear_embedding/linear_embedding/MatMul/ReadVariableOp:V R
'
_output_shapes
:���������5
'
_user_specified_namespecies_input
�#
�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1339

inputs'
linear_embedding_1317:5
identity��(linear_embedding/StatefulPartitionedCall�9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp�
(linear_embedding/StatefulPartitionedCallStatefulPartitionedCallinputslinear_embedding_1317*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1269�
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOplinear_embedding_1317*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOplinear_embedding_1317*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: �
IdentityIdentity1linear_embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^linear_embedding/StatefulPartitionedCall:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2T
(linear_embedding/StatefulPartitionedCall(linear_embedding/StatefulPartitionedCall2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�!
�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1269

inputs0
matmul_readvariableop_resource:5
identity��MatMul/ReadVariableOp�9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: _
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^MatMul/ReadVariableOp:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�!
�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1546

inputs0
matmul_readvariableop_resource:5
identity��MatMul/ReadVariableOp�9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: _
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^MatMul/ReadVariableOp:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�
�
/__inference_linear_embedding_layer_call_fn_1351
species_input
unknown:5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallspecies_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1339o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������5
'
_user_specified_namespecies_input
�#
�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1489

inputsA
/linear_embedding_matmul_readvariableop_resource:5
identity��&linear_embedding/MatMul/ReadVariableOp�9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp�
&linear_embedding/MatMul/ReadVariableOpReadVariableOp/linear_embedding_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
linear_embedding/MatMulMatMulinputs.linear_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOp/linear_embedding_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOp/linear_embedding_matmul_readvariableop_resource*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: p
IdentityIdentity!linear_embedding/MatMul:product:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^linear_embedding/MatMul/ReadVariableOp:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2P
&linear_embedding/MatMul/ReadVariableOp&linear_embedding/MatMul/ReadVariableOp2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�$
�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1401
species_input'
linear_embedding_1379:5
identity��(linear_embedding/StatefulPartitionedCall�9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp�
(linear_embedding/StatefulPartitionedCallStatefulPartitionedCallspecies_inputlinear_embedding_1379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1269�
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOplinear_embedding_1379*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOplinear_embedding_1379*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: �
IdentityIdentity1linear_embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^linear_embedding/StatefulPartitionedCall:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2T
(linear_embedding/StatefulPartitionedCall(linear_embedding/StatefulPartitionedCall2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp:V R
'
_output_shapes
:���������5
'
_user_specified_namespecies_input
�
}
"__inference_signature_wrapper_1430
species_input
unknown:5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallspecies_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *(
f#R!
__inference__wrapped_model_1237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������5
'
_user_specified_namespecies_input
�
�
/__inference_linear_embedding_layer_call_fn_1464

inputs
unknown:5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1339o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�
�
/__inference_linear_embedding_layer_call_fn_1521

inputs
unknown:5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1269o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������5
 
_user_specified_nameinputs
�
�
__inference__traced_save_1595
file_prefix6
2savev2_linear_embedding_kernel_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*i
value`B^B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_linear_embedding_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*!
_input_shapes
: :5: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:5:

_output_shapes
: 
�$
�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1376
species_input'
linear_embedding_1354:5
identity��(linear_embedding/StatefulPartitionedCall�9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp�<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp�
(linear_embedding/StatefulPartitionedCallStatefulPartitionedCallspecies_inputlinear_embedding_1354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*L
config_proto<:

CPU

GPU

XLA_CPU

XLA_GPU2 *0J 8� *S
fNRL
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1269�
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOpReadVariableOplinear_embedding_1354*
_output_shapes

:5*
dtype0�
2linear_embedding/kernel/Regularizer/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
-linear_embedding/kernel/Regularizer/transpose	TransposeDlinear_embedding/kernel/Regularizer/transpose/ReadVariableOp:value:0;linear_embedding/kernel/Regularizer/transpose/perm:output:0*
T0*
_output_shapes

:5�
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOpReadVariableOplinear_embedding_1354*
_output_shapes

:5*
dtype0�
*linear_embedding/kernel/Regularizer/MatMulMatMul1linear_embedding/kernel/Regularizer/transpose:y:0Alinear_embedding/kernel/Regularizer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:y
,linear_embedding/kernel/Regularizer/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  �?p
.linear_embedding/kernel/Regularizer/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : �
5linear_embedding/kernel/Regularizer/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
����������
5linear_embedding/kernel/Regularizer/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������
:linear_embedding/kernel/Regularizer/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,linear_embedding/kernel/Regularizer/eye/diagMatrixDiagV35linear_embedding/kernel/Regularizer/eye/ones:output:07linear_embedding/kernel/Regularizer/eye/diag/k:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_rows:output:0>linear_embedding/kernel/Regularizer/eye/diag/num_cols:output:0Clinear_embedding/kernel/Regularizer/eye/diag/padding_value:output:0*
T0*
_output_shapes

:�
'linear_embedding/kernel/Regularizer/subSub4linear_embedding/kernel/Regularizer/MatMul:product:05linear_embedding/kernel/Regularizer/eye/diag:output:0*
T0*
_output_shapes

:�
*linear_embedding/kernel/Regularizer/SquareSquare+linear_embedding/kernel/Regularizer/sub:z:0*
T0*
_output_shapes

:z
)linear_embedding/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
'linear_embedding/kernel/Regularizer/SumSum.linear_embedding/kernel/Regularizer/Square:y:02linear_embedding/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: �
(linear_embedding/kernel/Regularizer/SqrtSqrt0linear_embedding/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
)linear_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
'linear_embedding/kernel/Regularizer/mulMul2linear_embedding/kernel/Regularizer/mul/x:output:0,linear_embedding/kernel/Regularizer/Sqrt:y:0*
T0*
_output_shapes
: �
IdentityIdentity1linear_embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^linear_embedding/StatefulPartitionedCall:^linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp=^linear_embedding/kernel/Regularizer/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������5: 2T
(linear_embedding/StatefulPartitionedCall(linear_embedding/StatefulPartitionedCall2v
9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp9linear_embedding/kernel/Regularizer/MatMul/ReadVariableOp2|
<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp<linear_embedding/kernel/Regularizer/transpose/ReadVariableOp:V R
'
_output_shapes
:���������5
'
_user_specified_namespecies_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
species_input6
serving_default_species_input:0���������5D
linear_embedding0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�F
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
/__inference_linear_embedding_layer_call_fn_1297
/__inference_linear_embedding_layer_call_fn_1457
/__inference_linear_embedding_layer_call_fn_1464
/__inference_linear_embedding_layer_call_fn_1351�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
trace_1
trace_2
trace_32�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1489
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1514
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1376
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1401�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�B�
__inference__wrapped_model_1237species_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
 serving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
&trace_02�
/__inference_linear_embedding_layer_call_fn_1521�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z&trace_0
�
'trace_02�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1546�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z'trace_0
):'52linear_embedding/kernel
�
(trace_02�
__inference_loss_fn_0_1569�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z(trace_0
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_linear_embedding_layer_call_fn_1297species_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_linear_embedding_layer_call_fn_1457inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_linear_embedding_layer_call_fn_1464inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
/__inference_linear_embedding_layer_call_fn_1351species_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1489inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1514inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1376species_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1401species_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference_signature_wrapper_1430species_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_linear_embedding_layer_call_fn_1521inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1546inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_1569"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� �
__inference__wrapped_model_1237�6�3
,�)
'�$
species_input���������5
� "C�@
>
linear_embedding*�'
linear_embedding����������
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1376j>�;
4�1
'�$
species_input���������5
p 

 
� "%�"
�
0���������
� �
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1401j>�;
4�1
'�$
species_input���������5
p

 
� "%�"
�
0���������
� �
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1489c7�4
-�*
 �
inputs���������5
p 

 
� "%�"
�
0���������
� �
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1514c7�4
-�*
 �
inputs���������5
p

 
� "%�"
�
0���������
� �
J__inference_linear_embedding_layer_call_and_return_conditional_losses_1546[/�,
%�"
 �
inputs���������5
� "%�"
�
0���������
� �
/__inference_linear_embedding_layer_call_fn_1297]>�;
4�1
'�$
species_input���������5
p 

 
� "�����������
/__inference_linear_embedding_layer_call_fn_1351]>�;
4�1
'�$
species_input���������5
p

 
� "�����������
/__inference_linear_embedding_layer_call_fn_1457V7�4
-�*
 �
inputs���������5
p 

 
� "�����������
/__inference_linear_embedding_layer_call_fn_1464V7�4
-�*
 �
inputs���������5
p

 
� "�����������
/__inference_linear_embedding_layer_call_fn_1521N/�,
%�"
 �
inputs���������5
� "����������9
__inference_loss_fn_0_1569�

� 
� "� �
"__inference_signature_wrapper_1430�G�D
� 
=�:
8
species_input'�$
species_input���������5"C�@
>
linear_embedding*�'
linear_embedding���������