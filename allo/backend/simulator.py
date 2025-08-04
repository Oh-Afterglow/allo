# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, super-init-not-called, too-many-nested-blocks, too-many-branches
# pylint: disable=consider-using-enumerate, no-value-for-parameter, too-many-function-args, redefined-variable-type

import os
from ..backend.llvm import LLVMModule
from .._mlir.ir import (
    Location,
    UnitAttr,
    InsertionPoint,
    Module,
    Context,
    Region,
    RegionSequence,
    Block,
    BlockArgument,
    BlockArgumentList,
    OpView,
    OpResult,
    OpOperandList,
    Operation,
    Value,
    Type,
    TypeAttr,
    StringAttr,
    IntegerAttr,
    Attribute,
    ArrayAttr,
    DenseI8ArrayAttr,
    DenseI32ArrayAttr,
    DenseIntElementsAttr,
    FlatSymbolRefAttr,
    AffineMapAttr,
    AffineMap,
    AffineExpr,
    FunctionType,
    MemRefType,
    IntegerType,
    FloatType,
    IndexType,
    VectorType,
)
from .._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    memref as memref_d,
    openmp as openmp_d,
    async_dialect as async_d,
    arith as arith_d,
    index as index_d,
    affine as affine_d,
    vector as vector_d,
    scf as scf_d,
    cf as cf_d,
    llvm as llvm_d,
)
from .._mlir.passmanager import PassManager
from .._mlir.execution_engine import ExecutionEngine
from ..ir.transform import find_func_in_module
from ..passes import decompose_library_function
from ..utils import get_func_inputs_outputs


# The `walk` function
def recursive_collect_ops(
    top_op: Operation, target_op_type: tuple[type], res_list: list
):
    if isinstance(top_op, target_op_type):
        res_list.append(top_op)
    for region in top_op.regions:
        for block in region.blocks:
            for op in block:
                recursive_collect_ops(op, target_op_type, res_list)


# Useful when searching for omp operations after lowering
def recursive_collect_ops_by_name(
    top_op: Operation, target_op_name: str, res_list: list
):
    if top_op.name == target_op_name:
        res_list.append(top_op)
    for region in top_op.regions:
        for block in region.blocks:
            for op in block:
                recursive_collect_ops_by_name(op, target_op_name, res_list)


def collect_ops(module: Module, top_func_name: str):
    func = find_func_in_module(module, top_func_name)
    assert isinstance(func.body, Region)
    top_func_ops = func.body.blocks[0].operations
    pe_call_define_ops: dict[func_d.CallOp, func_d.FuncOp] = {}
    stream_construct_ops: dict[str, allo_d.StreamConstructOp] = {}
    for op in top_func_ops:
        if isinstance(op, memref_d.AllocOp):
            continue
        if isinstance(op, func_d.CallOp):
            callee_name = str(op.callee)[1:]
            if not callee_name.startswith(("load_buf", "store_res")):
                for mod_op in module.body.operations:
                    if isinstance(mod_op, func_d.FuncOp):
                        if callee_name == str(mod_op.sym_name).strip('"'):
                            pe_call_define_ops[op] = mod_op
                            break
        elif isinstance(op, allo_d.StreamConstructOp):
            stream_name = str(op.attributes["name"]).strip('"')
            stream_construct_ops[stream_name] = op
    return pe_call_define_ops, stream_construct_ops


def build_omp_dataflow_simulator(
    module: Module, top_func_name: str, use_task: bool = False
):
    with module.context, Location.unknown():
        pe_call_define_ops, stream_construct_ops = collect_ops(module, top_func_name)
        # Construct Memref variables for pipes
        stream_struct_table: dict[str, OpResult] = {}  # stream name: stream struct
        stream_type_table: dict[str, MemRefType] = {}

        const_0_defined = False
        empty_map = AffineMapAttr.get(AffineMap.get(0, 0, []))
        int_type = IntegerType.get_signless(32)
        memref_scalar_int_type = MemRefType.get([], int_type)
        for stream_access_op in stream_construct_ops.values():
            stream_name = stream_access_op.attributes["name"]
            stream_type = allo_d.StreamType(stream_access_op.result.type)
            stream_item_type = stream_type.base_type
            stream_depth = stream_type.depth
            assert isinstance(stream_item_type, (MemRefType, IntegerType, FloatType))
            assert isinstance(stream_depth, int)
            ip = InsertionPoint(beforeOperation=stream_access_op)
            if isinstance(stream_item_type, MemRefType):
                item_element_type = stream_item_type.element_type
                if not isinstance(item_element_type, (IntegerType, FloatType)):
                    raise NotImplementedError()
                memref_stream_type = MemRefType.get(
                    shape=[stream_depth + 1] + stream_item_type.shape,
                    element_type=item_element_type,
                )
            else:
                memref_stream_type = MemRefType.get(
                    shape=[stream_depth + 1], element_type=stream_item_type
                )
            stream_memref_op = memref_d.AllocOp(memref_stream_type, [], [], ip=ip)
            stream_head_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
            stream_tail_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
            if not const_0_defined:
                const_zero = arith_d.ConstantOp(int_type, 0, ip=ip)
                const_0_defined = True
            memref_d.StoreOp(value=const_zero, memref=stream_head_op, indices=[], ip=ip)
            memref_d.StoreOp(value=const_zero, memref=stream_tail_op, indices=[], ip=ip)
            fifo_struct_type = allo_d.StructType.get(
                members=[
                    memref_stream_type,
                    memref_scalar_int_type,
                    memref_scalar_int_type,
                ],
                context=module.context,
            )
            fifo_struct_op = allo_d.StructConstructOp(
                output=fifo_struct_type,
                input=[stream_memref_op, stream_head_op, stream_tail_op],
                ip=ip,
            )
            fifo_struct_memref_type = MemRefType.get([], fifo_struct_type)
            stream_memref_op = memref_d.AllocOp(fifo_struct_memref_type, [], [], ip=ip)
            stream_memref_op.attributes["name"] = stream_name
            affine_d.AffineStoreOp(
                value=fifo_struct_op,
                memref=stream_memref_op,
                indices=[],
                map=empty_map,
                ip=ip,
            )
            stream_name_str = str(stream_name).strip('"')
            stream_head_op.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_head"
            )
            stream_tail_op.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_tail"
            )
            stream_memref_op.attributes["name"] = stream_name
            stream_struct_table[stream_name_str] = stream_memref_op.result
            stream_type_table[stream_name_str] = memref_stream_type

        # Transfrom the stream operations in function calls
        for call_op, func_def_op in pe_call_define_ops.items():
            # Get the correspondence between arguments and passed pipes
            arg_stream_table: dict[BlockArgument, str] = {}  # arg: stream name
            assert isinstance(call_op.operands_, OpOperandList)
            assert isinstance(func_def_op.arguments, BlockArgumentList)
            assert len(call_op.operands_) == len(func_def_op.arguments)
            for i in range(len(call_op.operands_)):
                arg_instance = call_op.operands_[i]
                for stream_name, stream_construct_op in stream_construct_ops.items():
                    if Value(stream_construct_op.result) == arg_instance:
                        arg_def = func_def_op.arguments[i]
                        arg_stream_table[arg_def] = stream_name
            # Collect and replace `stream_get`s and `stream_put`s
            func_stream_ops = []
            recursive_collect_ops(
                func_def_op, (allo_d.StreamGetOp, allo_d.StreamPutOp), func_stream_ops
            )
            for stream_access_op in func_stream_ops:
                assert isinstance(
                    stream_access_op, (allo_d.StreamGetOp, allo_d.StreamPutOp)
                )
                replace_ip = InsertionPoint(beforeOperation=stream_access_op)
                # Have to leverage weak typing here
                stream = stream_access_op.stream
                stream_arg = BlockArgument(stream)
                stream_name = arg_stream_table[stream_arg]
                stream_type = stream_type_table[stream_name]
                stream_memref = stream_struct_table[stream_name]
                # Change argument definitions
                stream_arg.set_type(stream_memref.type)
                old_func_type = func_def_op.type
                new_inputs = old_func_type.inputs.copy()
                new_inputs[stream_arg.arg_number] = stream_memref.type
                new_func_type = FunctionType.get(
                    inputs=new_inputs,
                    results=old_func_type.results,
                    context=old_func_type.context,
                )
                func_def_op.attributes["function_type"] = TypeAttr.get(
                    new_func_type, module.context
                )
                call_op.operands_[stream_arg.arg_number] = stream_memref
                # FIFO access
                # Spin and wait for the FIFO to be not full
                assert isinstance(stream_memref.type, MemRefType)
                stream_struct = affine_d.AffineLoadOp(
                    result=stream_memref.type.element_type,
                    memref=stream_arg,
                    indices=[],
                    map=empty_map,
                    ip=replace_ip,
                )
                head_ptr = allo_d.StructGetOp(
                    output=memref_scalar_int_type,
                    input=stream_struct,
                    index=1,
                    ip=replace_ip,
                )
                tail_ptr = allo_d.StructGetOp(
                    output=memref_scalar_int_type,
                    input=stream_struct,
                    index=2,
                    ip=replace_ip,
                )
                fifo_ptr = allo_d.StructGetOp(
                    output=stream_type, input=stream_struct, index=0, ip=replace_ip
                )
                const_one = arith_d.ConstantOp(int_type, 1, ip=replace_ip)
                const_fifo_depth = arith_d.ConstantOp(
                    int_type, stream_type.get_dim_size(0), ip=replace_ip
                )
                if isinstance(stream_access_op, allo_d.StreamPutOp):
                    tail_val_op = memref_d.LoadOp(
                        memref=tail_ptr, indices=[], ip=replace_ip
                    )
                    tail_inc_op = arith_d.AddIOp(
                        lhs=tail_val_op, rhs=const_one, ip=replace_ip
                    )
                    tail_next_op = arith_d.RemUIOp(
                        lhs=tail_inc_op, rhs=const_fifo_depth, ip=replace_ip
                    )
                else:
                    assert isinstance(stream_access_op, allo_d.StreamGetOp)
                    head_val_op = memref_d.LoadOp(
                        memref=head_ptr, indices=[], ip=replace_ip
                    )
                    head_inc_op = arith_d.AddIOp(
                        lhs=head_val_op, rhs=const_one, ip=replace_ip
                    )
                    head_next_op = arith_d.RemUIOp(
                        lhs=head_inc_op, rhs=const_fifo_depth, ip=replace_ip
                    )
                spin_while_op = scf_d.WhileOp(results_=[], inits=[], ip=replace_ip)
                assert isinstance(spin_while_op.before, Region)
                assert isinstance(spin_while_op.after, Region)
                before_block = Block.create_at_start(
                    parent=spin_while_op.before, arg_types=[]
                )
                before_ip = InsertionPoint(before_block)
                openmp_d.FlushOp([], ip=before_ip)
                after_block = Block.create_at_start(
                    parent=spin_while_op.after, arg_types=[]
                )
                after_ip = InsertionPoint(after_block)
                openmp_d.TaskyieldOp(ip=after_ip)
                scf_d.YieldOp(results_=[], ip=after_ip)
                if isinstance(stream_access_op, allo_d.StreamPutOp):
                    head_val_op = memref_d.LoadOp(
                        memref=head_ptr, indices=[], ip=before_ip
                    )
                    cmp_op = arith_d.CmpIOp(
                        predicate=0, lhs=head_val_op, rhs=tail_next_op, ip=before_ip
                    )
                    scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
                    data = stream_access_op.data
                    assert isinstance(data, Value)  # Vector or scalar
                    tail_index_op = index_d.CastUOp(
                        output=IndexType.get(module.context),
                        input=tail_val_op,
                        ip=replace_ip,
                    )
                    if isinstance(data.type, MemRefType):  # Vector
                        # Data is an `alloc` pointer and should be loaded first
                        element_type = data.type.element_type
                        if not isinstance(element_type, (IntegerType, FloatType)):
                            # May get StructType involved in the future
                            raise NotImplementedError()
                        rank = data.type.rank
                        assert rank > 0
                        for_ip = replace_ip
                        for_induction_vars = []
                        for_ips: list[InsertionPoint] = (
                            []
                        )  # Reserved to insert affine.yield ops later
                        for i in range(rank):
                            dim_size = data.type.get_dim_size(i)
                            for_loop_op = affine_d.AffineForOp(0, dim_size, ip=for_ip)
                            for_induction_vars.append(for_loop_op.induction_variable)
                            for_ip = InsertionPoint(for_loop_op.body)
                            for_ips.append(for_ip)
                        element_dim_map = AffineMap.get(
                            dim_count=rank,
                            symbol_count=0,
                            exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                            context=module.context,
                        )
                        element_load_op = affine_d.AffineLoadOp(
                            result=element_type,
                            memref=data,
                            indices=for_induction_vars,
                            map=AffineMapAttr.get(element_dim_map),
                            ip=for_ip,
                        )  # Fetch the element
                        memref_d.StoreOp(
                            value=element_load_op,
                            memref=fifo_ptr,
                            indices=[tail_index_op] + for_induction_vars,
                            ip=for_ip,
                        )  # Put the element to the stream
                        for ip in for_ips:
                            affine_d.AffineYieldOp([], ip=ip)
                    else:  # Scalar
                        memref_d.StoreOp(
                            value=data,
                            memref=fifo_ptr,
                            indices=[tail_index_op],
                            ip=replace_ip,
                        )
                    # Atomic update of tail
                    critical_op = openmp_d.CriticalOp(ip=replace_ip)
                    critical_ip = InsertionPoint(
                        Block.create_at_start(critical_op.region)
                    )
                    memref_d.StoreOp(tail_next_op, tail_ptr, [], ip=critical_ip)
                    openmp_d.TerminatorOp(ip=critical_ip)
                else:
                    assert isinstance(stream_access_op, allo_d.StreamGetOp)
                    tail_val_op = memref_d.LoadOp(
                        memref=tail_ptr, indices=[], ip=before_ip
                    )
                    cmp_op = arith_d.CmpIOp(
                        0, lhs=head_val_op, rhs=tail_val_op, ip=before_ip
                    )
                    scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
                    orig_got_val = stream_access_op.res
                    assert isinstance(orig_got_val, OpResult)
                    head_index_op = index_d.CastUOp(
                        output=IndexType.get(module.context),
                        input=head_val_op,
                        ip=replace_ip,
                    )
                    if isinstance(orig_got_val.type, MemRefType):
                        element_type = orig_got_val.type.element_type
                        if not isinstance(element_type, (IntegerType, FloatType)):
                            raise NotImplementedError()
                        rank = orig_got_val.type.rank
                        assert rank > 0
                        # Create a memref for the loaded element
                        element_alloc_op = memref_d.AllocOp(
                            memref=orig_got_val.type,
                            dynamicSizes=[],
                            symbolOperands=[],
                            ip=replace_ip,
                        )
                        orig_got_val.replace_all_uses_with(element_alloc_op.result)
                        # Create the element load/store loop
                        for_ip = replace_ip
                        for_induction_vars = []
                        for_ips: list[InsertionPoint] = []
                        for i in range(rank):
                            for_loop_op = affine_d.AffineForOp(
                                0,
                                orig_got_val.type.get_dim_size(i),
                                ip=for_ip,
                            )
                            for_induction_vars.append(for_loop_op.induction_variable)
                            for_ip = InsertionPoint(for_loop_op.body)
                            for_ips.append(for_ip)
                        element_dim_map = AffineMap.get(
                            dim_count=rank,
                            symbol_count=0,
                            exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                            context=module.context,
                        )
                        element_load_op = memref_d.LoadOp(
                            memref=fifo_ptr,
                            indices=[head_index_op] + for_induction_vars,
                            ip=for_ip,  # The innermost Loop body
                        )
                        affine_d.AffineStoreOp(
                            value=element_load_op,
                            memref=element_alloc_op,
                            indices=for_induction_vars,
                            map=AffineMapAttr.get(element_dim_map),
                            ip=for_ip,
                        )
                        for ip in for_ips:
                            affine_d.AffineYieldOp([], ip=ip)
                    else:  # Scalar
                        new_get_op = memref_d.LoadOp(
                            memref=fifo_ptr, indices=[head_index_op], ip=replace_ip
                        )
                        orig_got_val.replace_all_uses_with(new_get_op.result)
                    critical_op = openmp_d.CriticalOp(ip=replace_ip)
                    critical_ip = InsertionPoint(
                        Block.create_at_start(critical_op.region)
                    )
                    memref_d.StoreOp(head_next_op, head_ptr, [], ip=critical_ip)
                    openmp_d.TerminatorOp(ip=critical_ip)
                stream_access_op.operation.erase()

        for op in stream_construct_ops.values():
            op.operation.erase()

        # Add the outmost `omp.parallel`
        assert len(pe_call_define_ops) > 0
        omp_ip = InsertionPoint(beforeOperation=list(pe_call_define_ops.keys())[0])
        omp_parallel_op = openmp_d.ParallelOp([], [], [], [], ip=omp_ip)
        assert isinstance(omp_parallel_op.region, Region)
        omp_parallel_block = Block.create_at_start(omp_parallel_op.region, [])

        if not use_task:
            # Add `omp.sections`
            ip_omp_parallel = InsertionPoint(omp_parallel_block)
            omp_sections_op = openmp_d.SectionsOp([], [], [], [], ip=ip_omp_parallel)
            omp_sections_block = Block.create_at_start(omp_sections_op.region, [])
            openmp_d.TerminatorOp(ip=ip_omp_parallel)

            # Add `omp.section`s for PE calls
            ip_omp_sections = InsertionPoint(omp_sections_block)
            for call_op in pe_call_define_ops:
                assert isinstance(call_op, OpView)
                omp_section_op = openmp_d.SectionOp(ip=ip_omp_sections)
                omp_section_block = Block.create_at_start(omp_section_op.region, [])
                ip_omp_section = InsertionPoint(omp_section_block)
                omp_term_op = openmp_d.TerminatorOp(ip=ip_omp_section)
                call_op.operation.move_before(omp_term_op.operation)
            openmp_d.TerminatorOp(ip=ip_omp_sections)
        else:
            # Add `omp.single` for the main thread
            ip_omp_parallel = InsertionPoint(omp_parallel_block)
            omp_single_op = openmp_d.SingleOp([], [], [], [], ip=ip_omp_parallel)
            omp_single_block = Block.create_at_start(omp_single_op.region, [])
            openmp_d.TerminatorOp(ip=ip_omp_parallel)

            # Add `omp.task` for PE calls
            ip_omp_single = InsertionPoint(omp_single_block)
            for call_op in pe_call_define_ops:
                assert isinstance(call_op, OpView)
                omp_task_op = openmp_d.TaskOp([], [], [], [], [], ip=ip_omp_single)
                omp_task_block = Block.create_at_start(omp_task_op.region, [])
                ip_omp_task = InsertionPoint(omp_task_block)
                omp_term_op = openmp_d.TerminatorOp(ip=ip_omp_task)
                call_op.operation.move_before(omp_term_op.operation)
            openmp_d.TerminatorOp(ip=ip_omp_single)


# This pass is only meant to run on fully lowered MLIR code
# Note: OpenMP operations in lowered IR are not the original operation types anymore
def convert_critical_write_to_atomic_write(module: Module):
    with module.context, Location.unknown():
        omp_critical_ops = []
        for op in module.body:
            if not isinstance(op, llvm_d.LLVMFuncOp):
                continue
            recursive_collect_ops_by_name(op, "omp.critical", omp_critical_ops)
        for critical_op in omp_critical_ops:
            # Transform a critical area with only the store op and omp.terminator
            assert isinstance(critical_op.regions, RegionSequence)
            if len(critical_op.regions) != 1:
                continue
            region = critical_op.regions[0]
            if len(region.blocks) != 1:
                continue
            block = region.blocks[0]
            if len(block.operations) != 2:
                continue
            if (
                not isinstance(block.operations[0], llvm_d.StoreOp)
                or block.operations[1].name != "omp.terminator"
            ):
                continue
            store_op = block.operations[0]
            assert isinstance(store_op, llvm_d.StoreOp)
            store_ip = InsertionPoint(critical_op)
            openmp_d.AtomicWriteOp(x=store_op.addr, expr=store_op.value, ip=store_ip)
            critical_op.operation.erase()


class LLVMOMPModule(LLVMModule):
    def __init__(self, mod: Module, top_func_name: str, ext_libs=None):
        with Context() as ctx:
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            ext_libs = [] if ext_libs is None else ext_libs
            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)
            self.module = decompose_library_function(self.module)

            build_omp_dataflow_simulator(self.module, self.top_func_name, use_task=True)
            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()

            # Start lowering
            # Lower linalg for AIE
            pm = PassManager.parse(
                "builtin.module("
                "one-shot-bufferize,"
                "expand-strided-metadata,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)
            # print(self.module)
            # Lower StructType
            allo_d.lower_composite_type(self.module)
            # Reference: https://discourse.llvm.org/t/help-lowering-affine-loop-to-openmp/72441/9
            pm = PassManager.parse(
                "builtin.module("
                "lower-affine,"
                "convert-scf-to-cf,"
                "finalize-memref-to-llvm,"
                "convert-func-to-llvm,"
                "convert-index-to-llvm,"
                "convert-cf-to-llvm,"
                "convert-openmp-to-llvm,"
                "canonicalize"
                ")"
            )
            pm.run(self.module.operation)
            convert_critical_write_to_atomic_write(self.module)

            assert os.getenv("LLVM_BUILD_DIR") is not None, "LLVM_BUILD_DIR is not set"
            shared_libs = [
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_runner_utils.so"
                ),
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_c_runner_utils.so"
                ),
                os.path.join(os.getenv("LLVM_BUILD_DIR"), "lib", "libomp.so"),
            ]
            shared_libs += [lib.compile_shared_lib() for lib in ext_libs]
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=2, shared_libs=shared_libs
            )


def build_async_dataflow_simulator(module: Module, top_func_name: str, debug=False):
    with module.context, Location.unknown():
        pe_call_define_ops, stream_construct_ops = collect_ops(module, top_func_name)

        # Debug: insert printf function declaration and global format string
        module_ip = InsertionPoint.at_block_begin(module.body)
        llvm_ptr_type = Type.parse("!llvm.ptr")
        string_type = Type.parse("!llvm.array<17 x i8>")
        printf_func_type = TypeAttr.parse("!llvm.func<i32 (ptr,...)>")
        llvm_d.LLVMFuncOp(
            sym_name=StringAttr.get("printf"),
            function_type=printf_func_type,
            ip=module_ip,
        )
        llvm_d.GlobalOp(
            global_type=string_type,
            sym_name=StringAttr.get(".str_producer_debug"),
            linkage=Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=StringAttr.get(b"Producer: %d %d\n\00"),
            ip=module_ip,
        )
        llvm_d.GlobalOp(
            global_type=string_type,
            sym_name=StringAttr.get(".str_consumer_debug"),
            linkage=Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=StringAttr.get(b"Consumer: %d %d\n\00"),
            ip=module_ip,
        )
        llvm_d.GlobalOp(
            global_type=string_type,
            sym_name=StringAttr.get(".str_prod_ref_debug"),
            linkage=Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=StringAttr.get(b"ProducerDropRef\n\00"),
            ip=module_ip,
        )
        llvm_d.GlobalOp(
            global_type=string_type,
            sym_name=StringAttr.get(".str_cons_ref_debug"),
            linkage=Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=StringAttr.get(b"ConsumerDropRef\n\00"),
            ip=module_ip,
        )

        # Construct Memref variables for pipes
        stream_struct_table: dict[str, OpResult] = {}  # stream name: stream struct
        stream_type_table: dict[str, MemRefType] = {}

        const_0_defined = False
        int32_type = IntegerType.get_signless(32)
        int64_type = IntegerType.get_signless(64)
        memref_scalar_int_type = MemRefType.get([], int32_type)
        async_token_type = Type.parse("!async.token")
        async_val_token_type = Type.parse("!async.value<!async.token>")
        # const_1_32_defined = False
        tokens = []  # Collect token ptrs for later cleaning up
        for stream_construct_op in stream_construct_ops.values():
            stream_name = stream_construct_op.attributes["name"]
            stream_type = allo_d.StreamType(stream_construct_op.result.type)
            stream_item_type = stream_type.base_type
            stream_depth = stream_type.depth
            assert isinstance(stream_item_type, (MemRefType, IntegerType, FloatType))
            assert isinstance(stream_depth, int)
            ip = InsertionPoint(beforeOperation=stream_construct_op)
            # if not const_1_32_defined:
            #     llvm_const_1_32 = llvm_d.ConstantOp(
            #         res=int32_type, value=IntegerAttr.get(int32_type, 1), ip=ip
            #     )
            #     const_1_32_defined = True
            if isinstance(stream_item_type, MemRefType):
                item_element_type = stream_item_type.element_type
                if not isinstance(item_element_type, (IntegerType, FloatType)):
                    raise NotImplementedError()
                memref_stream_type = MemRefType.get(
                    shape=[stream_depth + 1] + stream_item_type.shape,
                    element_type=item_element_type,
                )
            else:
                memref_stream_type = MemRefType.get(
                    shape=[stream_depth + 1], element_type=stream_item_type
                )
            stream_memref_op = memref_d.AllocOp(memref_stream_type, [], [], ip=ip)
            stream_head_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
            stream_tail_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
            stream_not_full = async_d.RuntimeCreateOp(async_val_token_type, ip=ip)
            stream_not_empty = async_d.RuntimeCreateOp(async_val_token_type, ip=ip)
            # stream_not_full = llvm_d.AllocaOp(res=llvm_ptr_type, arraySize=llvm_const_1_32, ip=ip)
            # stream_not_empty = llvm_d.AllocaOp(res=llvm_ptr_type, arraySize=llvm_const_1_32, ip=ip)
            tokens.append(stream_not_full)
            tokens.append(stream_not_empty)
            stream_not_full_init_token = async_d.RuntimeCreateOp(
                result=async_token_type, ip=ip
            )
            stream_not_empty_init_token = async_d.RuntimeCreateOp(
                result=async_token_type, ip=ip
            )
            async_d.RuntimeSetErrorOp(stream_not_full_init_token, ip=ip)
            # Stream is empty initially, so the not_empty flag is not ready
            async_d.RuntimeStoreOp(stream_not_full_init_token, stream_not_full, ip=ip)
            async_d.RuntimeSetAvailableOp(stream_not_full, ip=ip)
            async_d.RuntimeStoreOp(stream_not_empty_init_token, stream_not_empty, ip=ip)
            async_d.RuntimeSetAvailableOp(stream_not_empty, ip=ip)
            # llvm_d.StoreOp(value=stream_not_full_init_token, addr=stream_not_full, ip=ip)
            # llvm_d.StoreOp(value=stream_not_empty_init_token, addr=stream_not_empty, ip=ip)
            # Initialize head and tail value to 0
            if not const_0_defined:
                const_zero = arith_d.ConstantOp(int32_type, 0, ip=ip)
                const_0_defined = True
            memref_d.StoreOp(const_zero, stream_head_op, [], ip=ip)
            memref_d.StoreOp(const_zero, stream_tail_op, [], ip=ip)

            fifo_struct_type = allo_d.StructType.get(
                members=[
                    memref_stream_type,  # FIFO
                    memref_scalar_int_type,
                    memref_scalar_int_type,  # Head and tail
                    # llvm_ptr_type, llvm_ptr_type,
                    async_val_token_type,
                    async_val_token_type,  # Not full and not empty
                ],
                context=module.context,
            )
            fifo_struct_op = allo_d.StructConstructOp(
                output=fifo_struct_type,
                input=[
                    stream_memref_op,
                    stream_head_op,
                    stream_tail_op,
                    stream_not_full,
                    stream_not_empty,
                ],
                ip=ip,
            )
            stream_name_str = str(stream_name).strip('"')
            stream_head_op.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_head"
            )
            stream_tail_op.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_tail"
            )
            stream_not_full.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_not_full"
            )
            stream_not_empty.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_not_empty"
            )
            fifo_struct_op.attributes["name"] = stream_name
            stream_struct_table[stream_name_str] = fifo_struct_op.result
            stream_type_table[stream_name_str] = memref_stream_type

        # Transfrom the stream operations in function calls
        for call_op, func_def_op in pe_call_define_ops.items():
            # Get the correspondence between arguments and passed pipes
            arg_stream_table: dict[BlockArgument, str] = {}  # arg: stream name
            assert isinstance(call_op.operands_, OpOperandList)
            assert isinstance(func_def_op.arguments, BlockArgumentList)
            assert len(call_op.operands_) == len(func_def_op.arguments)
            for i in range(len(call_op.operands_)):
                arg_instance = call_op.operands_[i]
                for stream_name, stream_construct_op in stream_construct_ops.items():
                    if Value(stream_construct_op.result) == arg_instance:
                        arg_def = func_def_op.arguments[i]
                        arg_stream_table[arg_def] = stream_name
            # Collect and replace `stream_get`s and `stream_put`s
            func_stream_ops = []
            recursive_collect_ops(
                func_def_op, (allo_d.StreamGetOp, allo_d.StreamPutOp), func_stream_ops
            )
            for stream_access_op in func_stream_ops:
                assert isinstance(
                    stream_access_op, (allo_d.StreamGetOp, allo_d.StreamPutOp)
                )
                replace_ip = InsertionPoint(beforeOperation=stream_access_op)
                stream = stream_access_op.stream
                stream_arg = BlockArgument(stream)
                stream_name = arg_stream_table[stream_arg]
                stream_type = stream_type_table[stream_name]
                stream_memref = stream_struct_table[stream_name]
                # Change argument definitions
                stream_arg.set_type(stream_memref.type)
                old_func_type = func_def_op.type
                new_inputs = old_func_type.inputs.copy()
                new_inputs[stream_arg.arg_number] = stream_memref.type
                new_func_type = FunctionType.get(
                    inputs=new_inputs,
                    results=old_func_type.results,
                    context=old_func_type.context,
                )
                func_def_op.attributes["function_type"] = TypeAttr.get(
                    new_func_type, module.context
                )
                call_op.operands_[stream_arg.arg_number] = stream_memref

                # FIFO access
                # Await the async token for fullness/emptiness
                stream_struct = stream_arg
                head_ptr = allo_d.StructGetOp(
                    output=memref_scalar_int_type,
                    input=stream_struct,
                    index=1,
                    ip=replace_ip,
                )
                tail_ptr = allo_d.StructGetOp(
                    output=memref_scalar_int_type,
                    input=stream_struct,
                    index=2,
                    ip=replace_ip,
                )
                not_full_token_ptr = allo_d.StructGetOp(
                    output=async_val_token_type,
                    input=stream_struct,
                    index=3,
                    ip=replace_ip,
                )
                not_empty_token_ptr = allo_d.StructGetOp(
                    output=async_val_token_type,
                    input=stream_struct,
                    index=4,
                    ip=replace_ip,
                )
                fifo_ptr = allo_d.StructGetOp(
                    output=stream_type, input=stream_struct, index=0, ip=replace_ip
                )
                const_one = arith_d.ConstantOp(int32_type, 1, ip=replace_ip)
                const_fifo_depth = arith_d.ConstantOp(
                    int32_type, stream_type.get_dim_size(0), ip=replace_ip
                )
                if isinstance(stream_access_op, allo_d.StreamPutOp):
                    tail_val_op = memref_d.LoadOp(
                        memref=tail_ptr, indices=[], ip=replace_ip
                    )

                    # Debug: print head and tail before awaiting
                    # if debug:
                    #     head_val_dbg_op = memref_d.LoadOp(
                    #         memref=head_ptr, indices=[], ip=replace_ip
                    #     )
                    #     str_const_addr_op = llvm_d.AddressOfOp(
                    #         res=llvm_ptr_type,
                    #         global_name=FlatSymbolRefAttr.get(".str_producer_debug"),
                    #         ip=replace_ip,
                    #     )
                    #     str_addr_op = llvm_d.GEPOp(
                    #         res=llvm_ptr_type,
                    #         base=str_const_addr_op,
                    #         dynamicIndices=[],
                    #         rawConstantIndices=DenseI32ArrayAttr.get([0]),
                    #         elem_type=llvm_ptr_type,
                    #         ip=replace_ip,
                    #     )
                    #     llvm_d.CallOp(
                    #         result=int32_type,
                    #         callee_operands=[str_addr_op, head_val_dbg_op, tail_val_op],
                    #         op_bundle_operands=[],
                    #         op_bundle_sizes=DenseI32ArrayAttr.get([]),
                    #         op_bundle_tags=None,
                    #         callee=FlatSymbolRefAttr.get("printf"),
                    #         var_callee_type=printf_func_type,
                    #         ip=replace_ip,
                    #     )

                    # This replaces the spin wait
                    not_full_token = async_d.RuntimeLoadOp(
                        not_full_token_ptr, ip=replace_ip
                    )
                    async_d.RuntimeAwaitOp(operand=not_full_token, ip=replace_ip)
                    # Begin store
                    data = stream_access_op.data
                    assert isinstance(data, Value)
                    tail_index_op = index_d.CastUOp(
                        output=IndexType.get(module.context),
                        input=tail_val_op,
                        ip=replace_ip,
                    )
                    if isinstance(data.type, MemRefType):  # Vector
                        # Data is an `alloc` pointer and should be loaded first
                        element_type = data.type.element_type
                        if not isinstance(element_type, (IntegerType, FloatType)):
                            # May get StructType involved in the future
                            raise NotImplementedError()
                        rank = data.type.rank
                        assert rank > 0
                        for_ip = replace_ip
                        for_induction_vars = []
                        for_ips: list[InsertionPoint] = (
                            []
                        )  # Reserved to insert affine.yield ops later
                        for i in range(rank):
                            dim_size = data.type.get_dim_size(i)
                            for_loop_op = affine_d.AffineForOp(0, dim_size, ip=for_ip)
                            for_induction_vars.append(for_loop_op.induction_variable)
                            for_ip = InsertionPoint(for_loop_op.body)
                            for_ips.append(for_ip)
                        element_dim_map = AffineMap.get(
                            dim_count=rank,
                            symbol_count=0,
                            exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                            context=module.context,
                        )
                        element_load_op = affine_d.AffineLoadOp(
                            result=element_type,
                            memref=data,
                            indices=for_induction_vars,
                            map=AffineMapAttr.get(element_dim_map),
                            ip=for_ip,
                        )  # Fetch the element
                        memref_d.StoreOp(
                            value=element_load_op,
                            memref=fifo_ptr,
                            indices=[tail_index_op] + for_induction_vars,
                            ip=for_ip,
                        )  # Put the element to the stream
                        for ip in for_ips:
                            affine_d.AffineYieldOp([], ip=ip)
                    else:  # Scalar
                        memref_d.StoreOp(
                            value=data,
                            memref=fifo_ptr,
                            indices=[tail_index_op],
                            ip=replace_ip,
                        )
                    # End data store

                    # Check fullness
                    # new tail + 1 (old tail + 2)
                    const_two = arith_d.ConstantOp(int32_type, 2, ip=replace_ip)
                    tail_inc2_op = arith_d.AddIOp(
                        lhs=tail_val_op, rhs=const_two, ip=replace_ip
                    )
                    tail_next_next_val_op = arith_d.RemUIOp(
                        lhs=tail_inc2_op, rhs=const_fifo_depth, ip=replace_ip
                    )
                    # Load and compare head with new tail+1
                    head_load_op = memref_d.LoadOp(
                        memref=head_ptr, indices=[], ip=replace_ip
                    )
                    cmp_head_tail_op = arith_d.CmpIOp(
                        predicate=0,
                        lhs=head_load_op,
                        rhs=tail_next_next_val_op,
                        ip=replace_ip,
                    )
                    # If full, create a new token
                    # TODO: how to replace token atomically?
                    if_full_op = scf_d.IfOp(cond=cmp_head_tail_op, ip=replace_ip)
                    if_full_ip = InsertionPoint(if_full_op.then_block)
                    # Drop reference of the old token to destroy it
                    async_d.RuntimeDropRefOp(
                        operand=not_full_token,
                        count=IntegerAttr.get(int64_type, 1),
                        ip=if_full_ip,
                    )
                    new_full_token = async_d.RuntimeCreateOp(
                        async_token_type, ip=if_full_ip
                    )
                    async_d.RuntimeStoreOp(
                        new_full_token, not_full_token_ptr, ip=if_full_ip
                    )
                    scf_d.YieldOp(results_=[], ip=if_full_ip)

                    # Atomic update tail to make the change visible
                    update_tail_op = memref_d.GenericAtomicRMWOp(
                        result=int32_type, memref=tail_ptr, indices=[], ip=replace_ip
                    )
                    update_tail_block = Block.create_at_start(
                        update_tail_op.atomic_body, [int32_type]
                    )
                    update_tail_ip = InsertionPoint(update_tail_block)
                    tail_inc_op = arith_d.AddIOp(
                        lhs=update_tail_block.arguments[0],
                        rhs=const_one,
                        ip=update_tail_ip,
                    )
                    tail_mod_size_op = arith_d.RemUIOp(
                        lhs=tail_inc_op, rhs=const_fifo_depth, ip=update_tail_ip
                    )
                    memref_d.AtomicYieldOp(result=tail_mod_size_op, ip=update_tail_ip)

                    # Set non-empty token to ready (error)
                    # Abuse the error state here
                    not_empty_token = async_d.RuntimeLoadOp(
                        not_empty_token_ptr, ip=replace_ip
                    )
                    not_empty_ready = async_d.RuntimeIsErrorOp(
                        not_empty_token, ip=replace_ip
                    )
                    if_not_empty_ready_op = scf_d.IfOp(
                        cond=not_empty_ready, hasElse=True, ip=replace_ip
                    )
                    if_not_empty_ready_ip = InsertionPoint(
                        if_not_empty_ready_op.else_block
                    )
                    async_d.RuntimeSetErrorOp(
                        operand=not_empty_token, ip=if_not_empty_ready_ip
                    )
                    scf_d.YieldOp(results_=[], ip=if_not_empty_ready_ip)
                    scf_d.YieldOp(
                        results_=[], ip=InsertionPoint(if_not_empty_ready_op.then_block)
                    )
                else:  # stream_get
                    assert isinstance(stream_access_op, allo_d.StreamGetOp)
                    head_val_op = memref_d.LoadOp(head_ptr, [], ip=replace_ip)

                    # Debug: print head and tail before awaiting
                    # if debug:
                    #     tail_val_dbg_op = memref_d.LoadOp(tail_ptr, [], ip=replace_ip)
                    #     str_const_addr_op = llvm_d.AddressOfOp(
                    #         res=llvm_ptr_type,
                    #         global_name=FlatSymbolRefAttr.get(".str_consumer_debug"),
                    #         ip=replace_ip,
                    #     )
                    #     str_addr_op = llvm_d.GEPOp(
                    #         res=llvm_ptr_type,
                    #         base=str_const_addr_op,
                    #         dynamicIndices=[],
                    #         rawConstantIndices=DenseI32ArrayAttr.get([0]),
                    #         elem_type=llvm_ptr_type,
                    #         ip=replace_ip,
                    #     )
                    #     llvm_d.CallOp(
                    #         result=int32_type,
                    #         callee_operands=[str_addr_op, head_val_op, tail_val_dbg_op],
                    #         op_bundle_operands=[],
                    #         op_bundle_sizes=DenseI32ArrayAttr.get([]),
                    #         op_bundle_tags=None,
                    #         callee=FlatSymbolRefAttr.get("printf"),
                    #         var_callee_type=printf_func_type,
                    #         ip=replace_ip,
                    #     )

                    # Wait for not_empty
                    not_empty_token = async_d.RuntimeLoadOp(
                        not_empty_token_ptr, ip=replace_ip
                    )
                    async_d.RuntimeAwaitOp(operand=not_empty_token, ip=replace_ip)
                    orig_got_val = stream_access_op.res
                    assert isinstance(orig_got_val, OpResult)
                    head_index_op = index_d.CastUOp(
                        output=IndexType.get(module.context),
                        input=head_val_op,
                        ip=replace_ip,
                    )

                    # Begin load
                    if isinstance(orig_got_val.type, MemRefType):
                        element_type = orig_got_val.type.element_type
                        if not isinstance(element_type, (IntegerType, FloatType)):
                            raise NotImplementedError()
                        rank = orig_got_val.type.rank
                        assert rank > 0
                        # Create a memref for the loaded element
                        element_alloc_op = memref_d.AllocOp(
                            memref=orig_got_val.type,
                            dynamicSizes=[],
                            symbolOperands=[],
                            ip=replace_ip,
                        )
                        orig_got_val.replace_all_uses_with(element_alloc_op.result)
                        # Create the element load/store loop
                        for_ip = replace_ip
                        for_induction_vars = []
                        for_ips: list[InsertionPoint] = []
                        for i in range(rank):
                            for_loop_op = affine_d.AffineForOp(
                                0,
                                orig_got_val.type.get_dim_size(i),
                                ip=for_ip,
                            )
                            for_induction_vars.append(for_loop_op.induction_variable)
                            for_ip = InsertionPoint(for_loop_op.body)
                            for_ips.append(for_ip)
                        element_dim_map = AffineMap.get(
                            dim_count=rank,
                            symbol_count=0,
                            exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                            context=module.context,
                        )
                        element_load_op = memref_d.LoadOp(
                            memref=fifo_ptr,
                            indices=[head_index_op] + for_induction_vars,
                            ip=for_ip,  # The innermost Loop body
                        )
                        affine_d.AffineStoreOp(
                            value=element_load_op,
                            memref=element_alloc_op,
                            indices=for_induction_vars,
                            map=AffineMapAttr.get(element_dim_map),
                            ip=for_ip,
                        )
                        for ip in for_ips:
                            affine_d.AffineYieldOp([], ip=ip)
                    else:  # Scalar
                        new_get_op = memref_d.LoadOp(
                            memref=fifo_ptr, indices=[head_index_op], ip=replace_ip
                        )
                        orig_got_val.replace_all_uses_with(new_get_op.result)
                    # End load (not visible yet)

                    # Check emptiness, read tail
                    # Load and compare tail with head
                    head_inc_1_op = arith_d.AddIOp(
                        lhs=head_val_op, rhs=const_one, ip=replace_ip
                    )
                    head_mod_size_op = arith_d.RemUIOp(
                        lhs=head_inc_1_op, rhs=const_fifo_depth, ip=replace_ip
                    )
                    tail_load_op = memref_d.LoadOp(
                        memref=tail_ptr, indices=[], ip=replace_ip
                    )
                    check_empty_op = arith_d.CmpIOp(
                        predicate=0,
                        lhs=tail_load_op,
                        rhs=head_mod_size_op,
                        ip=replace_ip,
                    )  # Empty: tail == new head
                    # If empty, create a new token to block the next get
                    # TODO: atomic
                    if_empty_op = scf_d.IfOp(cond=check_empty_op, ip=replace_ip)
                    if_empty_ip = InsertionPoint(if_empty_op.then_block)
                    if debug:
                        str_const_addr_op = llvm_d.AddressOfOp(
                            res=llvm_ptr_type,
                            global_name=FlatSymbolRefAttr.get(".str_cons_ref_debug"),
                            ip=if_empty_ip,
                        )
                        str_addr_op = llvm_d.GEPOp(
                            res=llvm_ptr_type,
                            base=str_const_addr_op,
                            dynamicIndices=[],
                            rawConstantIndices=DenseI32ArrayAttr.get([0]),
                            elem_type=llvm_ptr_type,
                            ip=if_empty_ip,
                        )
                        llvm_d.CallOp(
                            result=int32_type,
                            callee_operands=[str_addr_op],
                            op_bundle_operands=[],
                            op_bundle_sizes=DenseI32ArrayAttr.get([]),
                            op_bundle_tags=None,
                            callee=FlatSymbolRefAttr.get("printf"),
                            var_callee_type=printf_func_type,
                            ip=if_empty_ip,
                        )
                    async_d.RuntimeDropRefOp(
                        operand=not_empty_token,
                        count=IntegerAttr.get(int64_type, 1),
                        ip=if_empty_ip,
                    )
                    new_empty_token = async_d.RuntimeCreateOp(
                        async_token_type, ip=if_empty_ip
                    )
                    async_d.RuntimeStoreOp(
                        new_empty_token, not_empty_token_ptr, ip=if_empty_ip
                    )
                    scf_d.YieldOp(results_=[], ip=if_empty_ip)

                    # Update head to make update visible
                    update_head_op = memref_d.GenericAtomicRMWOp(
                        result=int32_type, memref=head_ptr, indices=[], ip=replace_ip
                    )
                    update_head_block = Block.create_at_start(
                        parent=update_head_op.atomic_body, arg_types=[int32_type]
                    )
                    update_head_ip = InsertionPoint(update_head_block)
                    head_inc_1_op = arith_d.AddIOp(
                        lhs=update_head_block.arguments[0],
                        rhs=const_one,
                        ip=update_head_ip,
                    )
                    head_mod_size_op = arith_d.RemUIOp(
                        lhs=head_inc_1_op, rhs=const_fifo_depth, ip=update_head_ip
                    )
                    memref_d.AtomicYieldOp(result=head_mod_size_op, ip=update_head_ip)

                    # Set not_full_token to ready (error)
                    not_full_token = async_d.RuntimeLoadOp(
                        not_full_token_ptr, ip=replace_ip
                    )
                    not_full_ready = async_d.RuntimeIsErrorOp(
                        operand=not_full_token, ip=replace_ip
                    )
                    if_not_full_ready_op = scf_d.IfOp(
                        cond=not_full_ready, hasElse=True, ip=replace_ip
                    )  # Else branch is for not ready
                    if_not_full_ready_ip = InsertionPoint(
                        if_not_full_ready_op.else_block
                    )
                    async_d.RuntimeSetErrorOp(
                        operand=not_full_token, ip=if_not_full_ready_ip
                    )
                    scf_d.YieldOp(
                        [], ip=InsertionPoint(if_not_full_ready_op.then_block)
                    )
                    scf_d.YieldOp([], ip=if_not_full_ready_ip)
                stream_access_op.operation.erase()
        for op in stream_construct_ops.values():
            op.operation.erase()

        # Wrap calls with async.execute
        # This is a bit different from the omp one
        # A dummy operation is needed to mark the position before the PE call operation
        # If the call operation is directly referenced by an IP,
        # then it'll not be movable
        assert len(pe_call_define_ops) > 0
        call_ip = InsertionPoint(beforeOperation=list(pe_call_define_ops.keys())[0])

        dummy_op = arith_d.ConstantOp(int32_type, 1, ip=call_ip)
        call_ip = InsertionPoint(beforeOperation=dummy_op)
        num_pe_const = arith_d.ConstantOp(
            result=IndexType.get(), value=len(pe_call_define_ops), ip=call_ip
        )
        group_op = async_d.CreateGroupOp(size=num_pe_const, ip=call_ip)
        # do nothing here
        dummy_execute_op = async_d.ExecuteOp(async_token_type, [], [], [], ip=call_ip)
        dummy_execute_block = Block.create_at_start(dummy_execute_op.bodyRegion)
        dummy_ip = InsertionPoint(dummy_execute_block)
        async_d.YieldOp([], ip=dummy_ip)
        execute_tokens = [dummy_execute_op.token]
        for call_op in pe_call_define_ops.keys():
            assert isinstance(call_op, OpView)
            assert isinstance(call_op.operation, Operation)
            async_execute_op = async_d.ExecuteOp(
                async_token_type, [], [], [], ip=call_ip
            )
            assert isinstance(async_execute_op.bodyRegion, Region)
            async_execute_block = Block.create_at_start(async_execute_op.bodyRegion, [])
            execute_ip = InsertionPoint(async_execute_block)
            yield_op = async_d.YieldOp(operands_=[], ip=execute_ip)
            call_op.operation.move_before(yield_op)
            async_d.AddToGroupOp(async_execute_op, group_op, ip=call_ip)
            execute_tokens.append(async_execute_op.token)
        async_d.AwaitAllOp(group_op, ip=call_ip)

        # Destroy all tokens after all calls have returned
        for token_ptr in tokens:
            assert isinstance(token_ptr, async_d.RuntimeCreateOp)
            left_token = async_d.RuntimeLoadOp(storage=token_ptr, ip=call_ip)
            ready = async_d.RuntimeIsErrorOp(left_token, ip=call_ip)
            if_ready_op = scf_d.IfOp(cond=ready, hasElse=True, ip=call_ip)
            ready_ip = InsertionPoint(if_ready_op.then_block)
            not_ready_ip = InsertionPoint(if_ready_op.else_block)
            # if the token is not ready, the reference count will be 2
            async_d.RuntimeDropRefOp(
                operand=left_token, count=IntegerAttr.get(int64_type, 1), ip=ready_ip
            )
            scf_d.YieldOp(results_=[], ip=ready_ip)
            async_d.RuntimeDropRefOp(
                operand=left_token,
                count=IntegerAttr.get(int64_type, 2),
                ip=not_ready_ip,
            )
            scf_d.YieldOp(results_=[], ip=not_ready_ip)
            # After dropping the token, also drop the wrapper value
            async_d.RuntimeDropRefOp(
                operand=token_ptr, count=IntegerAttr.get(int64_type, 1), ip=call_ip
            )
        for token in execute_tokens:
            async_d.RuntimeDropRefOp(
                operand=token, count=IntegerAttr.get(int64_type, 1), ip=call_ip
            )
        async_d.RuntimeDropRefOp(
            operand=group_op, count=IntegerAttr.get(int64_type, 1), ip=call_ip
        )
        dummy_op.operation.erase()


class LLVMAsyncModule(LLVMModule):
    def __init__(self, mod: Module, top_func_name: str, ext_libs=None):
        with Context() as ctx:
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            ext_libs = [] if ext_libs is None else ext_libs
            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)
            self.module = decompose_library_function(self.module)

            build_async_dataflow_simulator(self.module, self.top_func_name)
            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()
            # print(self.module)

            # Start lowering
            # Lower linalg for AIE
            pm = PassManager.parse(
                "builtin.module("
                "one-shot-bufferize,"
                "expand-strided-metadata,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)
            # Lower StructType
            allo_d.lower_composite_type(self.module)
            pm = PassManager.parse(
                "builtin.module("
                "lower-affine,"
                "convert-scf-to-cf,"
                "async-to-async-runtime,"
                "convert-async-to-llvm,"
                "convert-func-to-llvm,"
                "convert-index-to-llvm,"
                "convert-cf-to-llvm,"
                "finalize-memref-to-llvm,"
                "canonicalize"
                ")"
            )
            pm.run(self.module.operation)
            # print(self.module)

            assert os.getenv("LLVM_BUILD_DIR") is not None, "LLVM_BUILD_DIR is not set"
            shared_libs = [
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_runner_utils.so"
                ),
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_c_runner_utils.so"
                ),
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_async_runtime.so"
                ),
            ]
            shared_libs += [lib.compile_shared_lib() for lib in ext_libs]
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=2, shared_libs=shared_libs
            )


def build_coroutine_dataflow_simulator(module: Module, top_func_name: str):
    with module.context, Location.unknown():
        pe_call_define_ops, stream_construct_ops = collect_ops(module, top_func_name)

        # Debug: insert printf function declaration and global format string
        module_ip = InsertionPoint.at_block_begin(module.body)
        llvm_ptr_type = Type.parse("!llvm.ptr")
        string_type = Type.parse("!llvm.array<17 x i8>")
        printf_func_type = TypeAttr.parse("!llvm.func<i32 (ptr,...)>")
        llvm_d.LLVMFuncOp(
            sym_name=StringAttr.get("printf"),
            function_type=printf_func_type,
            ip=module_ip,
        )
        llvm_d.GlobalOp(
            global_type=string_type,
            sym_name=StringAttr.get(".str_producer_debug"),
            linkage=Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=StringAttr.get(b"Producer: %d %d\n\00"),
            ip=module_ip,
        )
        llvm_d.GlobalOp(
            global_type=string_type,
            sym_name=StringAttr.get(".str_consumer_debug"),
            linkage=Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=StringAttr.get(b"Consumer: %d %d\n\00"),
            ip=module_ip,
        )
        llvm_d.GlobalOp(
            global_type=string_type,
            sym_name=StringAttr.get(".str_producer_block_debug"),
            linkage=Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=StringAttr.get(b"Producer block \n\00"),
            ip=module_ip,
        )
        llvm_d.GlobalOp(
            global_type=string_type,
            sym_name=StringAttr.get(".str_consumer_block_debug"),
            linkage=Attribute.parse("#llvm.linkage<private>"),
            constant=True,
            value=StringAttr.get(b"Consumer block \n\00"),
            ip=module_ip,
        )

        module_ip = InsertionPoint.at_block_begin(module.body)
        llvm_ptr_type = Type.parse("!llvm.ptr")
        alloc_func_type = TypeAttr.parse("!llvm.func<ptr (i64, i64)>")
        free_func_type = TypeAttr.parse("!llvm.func<void (ptr)>")
        runtime_exec_func_type = TypeAttr.parse("!llvm.func<void (ptr, ptr)>")
        coro_resume_func_type = free_func_type
        llvm_d.LLVMFuncOp(
            sym_name=StringAttr.get("aligned_alloc"),
            function_type=alloc_func_type,
            ip=module_ip,
        )
        llvm_d.LLVMFuncOp(
            sym_name=StringAttr.get("free"), function_type=free_func_type, ip=module_ip
        )
        llvm_d.LLVMFuncOp(
            sym_name=StringAttr.get("mlirAsyncRuntimeExecute"),
            function_type=runtime_exec_func_type,
            sym_visibility="private",
            ip=module_ip,
        )
        resume_func = llvm_d.LLVMFuncOp(
            sym_name=StringAttr.get("__resume"),
            function_type=coro_resume_func_type,
            sym_visibility="private",
            ip=module_ip,
        )
        assert isinstance(resume_func.body, Region)
        resume_func_body = Block.create_at_start(resume_func.body, [llvm_ptr_type])
        resume_ip = InsertionPoint(resume_func_body)
        llvm_d.CoroResumeOp(
            handle=resume_func.body.blocks[0].arguments[0], ip=resume_ip
        )
        llvm_d.ReturnOp(ip=resume_ip)

        # Construct Memref variables for pipes
        stream_struct_table: dict[str, OpResult] = {}  # stream name: stream struct
        stream_type_table: dict[str, MemRefType] = {}

        const_0_defined = False
        int32_type = IntegerType.get_signless(32)
        int64_type = IntegerType.get_signless(64)
        memref_scalar_int_type = MemRefType.get([], int32_type)
        # Transform the stream definitions in the top function
        for stream_access_op in stream_construct_ops.values():
            stream_name = stream_access_op.attributes["name"]
            stream_type = allo_d.StreamType(stream_access_op.result.type)
            stream_item_type = stream_type.base_type
            stream_depth = stream_type.depth
            assert isinstance(stream_item_type, (MemRefType, IntegerType, FloatType))
            assert isinstance(stream_depth, int)
            ip = InsertionPoint(beforeOperation=stream_access_op)
            if isinstance(stream_item_type, MemRefType):
                item_element_type = stream_item_type.element_type
                if not isinstance(item_element_type, (IntegerType, FloatType)):
                    raise NotImplementedError()
                memref_stream_type = MemRefType.get(
                    shape=[stream_depth + 1] + stream_item_type.shape,
                    element_type=item_element_type,
                )
            else:
                memref_stream_type = MemRefType.get(
                    shape=[stream_depth + 1], element_type=stream_item_type
                )
            stream_memref_op = memref_d.AllocOp(memref_stream_type, [], [], ip=ip)
            stream_head_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)
            stream_tail_op = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=ip)

            # Initialize head and tail value to 0
            if not const_0_defined:
                const_zero = arith_d.ConstantOp(int32_type, 0, ip=ip)
                const_0_defined = True
            memref_d.StoreOp(const_zero, stream_head_op, [], ip=ip)
            memref_d.StoreOp(const_zero, stream_tail_op, [], ip=ip)

            # Create structs
            fifo_struct_type = allo_d.StructType.get(
                members=[
                    memref_stream_type,  # FIFO
                    memref_scalar_int_type,
                    memref_scalar_int_type,  # Head and tail
                ],
                context=module.context,
            )
            fifo_struct_op = allo_d.StructConstructOp(
                output=fifo_struct_type,
                input=[
                    stream_memref_op,
                    stream_head_op,
                    stream_tail_op,
                ],
                ip=ip,
            )
            stream_name_str = str(stream_name).strip('"')
            stream_head_op.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_head"
            )
            stream_tail_op.attributes["name"] = StringAttr.get(
                f"{stream_name_str}_tail"
            )
            fifo_struct_op.attributes["name"] = stream_name
            stream_struct_table[stream_name_str] = fifo_struct_op.result
            stream_type_table[stream_name_str] = memref_stream_type

        # Transfrom the stream operations in function calls
        llvm_token_type = Type.parse("!llvm.token")
        bool_type = IntegerType.get_signless(1)
        int8_type = IntegerType.get_signless(8)
        for call_op, func_def_op in pe_call_define_ops.items():
            # Change function type, add the coro handle return value
            func_type = func_def_op.type
            coro_func_type = FunctionType.get(
                inputs=func_type.inputs.copy(),
                results=[llvm_ptr_type],
                context=func_type.context,
            )
            func_def_op.attributes["function_type"] = TypeAttr.get(
                coro_func_type, module.context
            )
            func_def_op.attributes["passthrough"] = ArrayAttr.get(
                [StringAttr.get("presplitcoroutine")]
            )
            # Initialize the coroutine at the beginning
            assert isinstance(func_def_op.body, Region)
            assert isinstance(func_def_op.arguments, BlockArgumentList)
            func_begin_block = func_def_op.body.blocks[0]
            func_body_block = Block.create_after(func_begin_block)
            # Move the original function body to the new block
            for op in func_begin_block.operations:
                if not isinstance(op, func_d.ReturnOp):
                    func_body_block.append(op)
                else:  # Remove the original return op
                    op.operation.erase()
            func_begin_ip = InsertionPoint(func_begin_block)
            llvm_const_0_32 = llvm_d.ConstantOp(
                res=int32_type, value=IntegerAttr.get(int32_type, 0), ip=func_begin_ip
            )
            llvm_const_0_64 = llvm_d.ConstantOp(
                res=int64_type, value=IntegerAttr.get(int64_type, 0), ip=func_begin_ip
            )
            llvm_const_0_8 = llvm_d.ConstantOp(
                res=int8_type, value=IntegerAttr.get(int8_type, 0), ip=func_begin_ip
            )
            llvm_const_1_32 = llvm_d.ConstantOp(
                res=int32_type, value=IntegerAttr.get(int32_type, 1), ip=func_begin_ip
            )
            llvm_const_minus1_32 = llvm_d.ConstantOp(
                res=int32_type, value=IntegerAttr.get(int32_type, -1), ip=func_begin_ip
            )
            llvm_const_1_64 = llvm_d.ConstantOp(
                res=int64_type, value=IntegerAttr.get(int64_type, 1), ip=func_begin_ip
            )
            llvm_const_false = llvm_d.ConstantOp(
                res=bool_type, value=IntegerAttr.get(bool_type, 0), ip=func_begin_ip
            )
            llvm_const_true = llvm_d.ConstantOp(
                res=bool_type, value=IntegerAttr.get(bool_type, 1), ip=func_begin_ip
            )
            llvm_null_ptr = llvm_d.ZeroOp(res=llvm_ptr_type, ip=func_begin_ip)
            promise_op = llvm_d.AllocaOp(
                res=llvm_ptr_type,
                arraySize=llvm_const_1_32,
                elem_type=TypeAttr.get(int32_type),
                ip=func_begin_ip,
            )
            llvm_d.StoreOp(value=llvm_const_0_32, addr=promise_op, ip=func_begin_ip)
            llvm_coro_id_op = llvm_d.CoroIdOp(
                res=llvm_token_type,
                align=llvm_const_0_32,
                promise=promise_op,
                coroaddr=llvm_null_ptr,
                fnaddrs=llvm_null_ptr,
                ip=func_begin_ip,
            )
            llvm_coro_size_op = llvm_d.CoroSizeOp(res=int64_type, ip=func_begin_ip)
            llvm_coro_align_op = llvm_d.CoroAlignOp(res=int64_type, ip=func_begin_ip)
            size_plus_align_op = llvm_d.AddOp(
                lhs=llvm_coro_size_op,
                rhs=llvm_coro_align_op,
                overflowFlags=None,
                ip=func_begin_ip,
            )
            minus_1_op = llvm_d.SubOp(
                lhs=size_plus_align_op,
                rhs=llvm_const_1_64,
                overflowFlags=None,
                ip=func_begin_ip,
            )
            minus_align_op = llvm_d.SubOp(
                lhs=llvm_const_0_64,
                rhs=llvm_coro_align_op,
                overflowFlags=None,
                ip=func_begin_ip,
            )
            and_op = llvm_d.AndOp(
                lhs=minus_1_op,
                rhs=minus_align_op,
                ip=func_begin_ip,
            )
            alloc_op = llvm_d.CallOp(
                result=llvm_ptr_type,
                callee_operands=[llvm_coro_align_op, and_op],
                op_bundle_operands=[],
                op_bundle_sizes=DenseI32ArrayAttr.get([]),
                op_bundle_tags=None,
                callee=FlatSymbolRefAttr.get("aligned_alloc"),
                # var_callee_type=alloc_func_type,
                ip=func_begin_ip,
            )
            llvm_coro_begin_op = llvm_d.CoroBeginOp(
                res=llvm_ptr_type, token=llvm_coro_id_op, mem=alloc_op, ip=func_begin_ip
            )  # handle
            coro_handle = llvm_coro_begin_op.res
            llvm_coro_save_op = llvm_d.CoroSaveOp(
                res=llvm_token_type, handle=coro_handle, ip=func_begin_ip
            )  # state
            # llvm_coro_save_op = llvm_d.NoneTokenOp(ip=func_begin_ip)
            first_suspend_op = llvm_d.CoroSuspendOp(
                res=int8_type,
                save=llvm_coro_save_op,
                final=llvm_const_false,
                ip=func_begin_ip,
            )
            # Add the final suspend point at the end of the task
            func_end_ip = InsertionPoint(func_body_block)
            llvm_coro_final_save_op = llvm_d.CoroSaveOp(
                res=llvm_token_type, handle=coro_handle, ip=func_end_ip
            )  # State (final)
            # llvm_coro_final_save_op = llvm_d.NoneTokenOp(ip=func_end_ip)
            final_suspend_op = llvm_d.CoroSuspendOp(
                res=int8_type,
                save=llvm_coro_final_save_op,
                final=llvm_const_true,
                ip=func_end_ip,
            )
            final_branch_cond = arith_d.CmpIOp(
                predicate=5,  # sge
                lhs=final_suspend_op,
                rhs=llvm_const_0_8,
                ip=func_end_ip,
            )
            # Create successor block "Cleanup", actually unreachable
            cleanup_block = Block.create_after(func_body_block)
            cleanup_ip = InsertionPoint(cleanup_block)
            # Create successor block "Suspend"
            suspend_block = Block.create_after(cleanup_block)
            suspend_ip = InsertionPoint(suspend_block)
            # Create a special "Suspend" block for the final suspend
            final_suspend_block = Block.create_after(suspend_block)
            final_suspend_ip = InsertionPoint(final_suspend_block)
            # Successor of suspend ops
            cf_d.SwitchOp(
                flag=first_suspend_op,
                defaultOperands=[],
                defaultDestination=suspend_block,
                case_values=DenseIntElementsAttr.get(
                    [IntegerAttr.get(int8_type, 0), IntegerAttr.get(int8_type, 1)]
                ),
                caseOperands=[],
                case_operand_segments=DenseI32ArrayAttr.get([0, 0]),
                caseDestinations=[func_body_block, cleanup_block],
                ip=func_begin_ip,
            )
            cf_d.CondBranchOp(
                condition=final_branch_cond,
                trueDestOperands=[],
                falseDestOperands=[],
                trueDest=cleanup_block,
                falseDest=final_suspend_block,
                ip=func_end_ip,
            )
            # Operations in "Cleanup"
            cleanup_free_op = llvm_d.CoroFreeOp(
                res=llvm_ptr_type, id=llvm_coro_id_op, handle=coro_handle, ip=cleanup_ip
            )
            llvm_d.CallOp(
                result=None,
                callee_operands=[cleanup_free_op],
                op_bundle_operands=[],
                op_bundle_sizes=DenseI32ArrayAttr.get([]),
                op_bundle_tags=None,
                callee=FlatSymbolRefAttr.get("free"),
                ip=cleanup_ip,
            )
            cf_d.BranchOp(destOperands=[], dest=suspend_block, ip=cleanup_ip)
            # Operations in "Suspend"
            llvm_none_op = llvm_d.NoneTokenOp(ip=suspend_ip)
            llvm_d.CoroEndOp(
                res=IntegerType.get_signless(1),
                handle=coro_handle,
                unwind=llvm_const_false,
                retvals=llvm_none_op,
                ip=suspend_ip,
            )  # Value unused
            func_d.ReturnOp(operands_=[coro_handle], ip=suspend_ip)
            # Operations in final "Suspend"
            llvm_d.StoreOp(
                value=llvm_const_minus1_32, addr=promise_op, ip=final_suspend_ip
            )
            cf_d.BranchOp(destOperands=[], dest=suspend_block, ip=final_suspend_ip)

            # Get the correspondence between arguments and passed pipes
            arg_stream_table: dict[BlockArgument, str] = {}  # arg: stream name
            assert isinstance(call_op.operands_, OpOperandList)
            assert isinstance(func_def_op.arguments, BlockArgumentList)
            assert len(call_op.operands_) == len(func_def_op.arguments)
            for i in range(len(call_op.operands_)):
                arg_instance = call_op.operands_[i]
                for stream_name, stream_construct_op in stream_construct_ops.items():
                    if Value(stream_construct_op.result) == arg_instance:
                        arg_def = func_def_op.arguments[i]
                        arg_stream_table[arg_def] = stream_name

            # Collect and replace `stream_get`s and `stream_put`s
            func_stream_ops = []
            recursive_collect_ops(
                func_def_op, (allo_d.StreamGetOp, allo_d.StreamPutOp), func_stream_ops
            )
            for stream_access_op in func_stream_ops:
                assert isinstance(
                    stream_access_op, (allo_d.StreamGetOp, allo_d.StreamPutOp)
                )
                replace_ip = InsertionPoint(beforeOperation=stream_access_op)
                stream = stream_access_op.stream
                stream_arg = BlockArgument(stream)
                stream_name = arg_stream_table[stream_arg]
                stream_type = stream_type_table[stream_name]
                stream_memref = stream_struct_table[stream_name]
                # Change argument definitions
                stream_arg.set_type(stream_memref.type)
                old_func_type = func_def_op.type
                new_inputs = old_func_type.inputs.copy()
                new_inputs[stream_arg.arg_number] = stream_memref.type
                new_func_type = FunctionType.get(
                    inputs=new_inputs,
                    results=old_func_type.results,
                    context=old_func_type.context,
                )
                func_def_op.attributes["function_type"] = TypeAttr.get(
                    new_func_type, module.context
                )
                call_op.operands_[stream_arg.arg_number] = stream_memref

                # Begin FIFO access
                # Get pointers needed
                stream_struct = stream_arg
                head_ptr = allo_d.StructGetOp(
                    output=memref_scalar_int_type,
                    input=stream_struct,
                    index=1,
                    ip=replace_ip,
                )
                tail_ptr = allo_d.StructGetOp(
                    output=memref_scalar_int_type,
                    input=stream_struct,
                    index=2,
                    ip=replace_ip,
                )
                fifo_ptr = allo_d.StructGetOp(
                    output=stream_type, input=stream_struct, index=0, ip=replace_ip
                )
                const_one = arith_d.ConstantOp(int32_type, 1, ip=replace_ip)
                const_fifo_depth = arith_d.ConstantOp(
                    int32_type, stream_type.get_dim_size(0), ip=replace_ip
                )
                # Calculate the next pointer update first
                if isinstance(stream_access_op, allo_d.StreamPutOp):
                    # Suspend if the FIFO is full
                    tail_val_op = memref_d.LoadOp(
                        memref=tail_ptr, indices=[], ip=replace_ip
                    )
                    tail_inc_op = arith_d.AddIOp(
                        lhs=tail_val_op, rhs=const_one, ip=replace_ip
                    )
                    tail_next_op = arith_d.RemUIOp(
                        lhs=tail_inc_op, rhs=const_fifo_depth, ip=replace_ip
                    )
                else:
                    assert isinstance(stream_access_op, allo_d.StreamGetOp)
                    head_val_op = memref_d.LoadOp(head_ptr, [], ip=replace_ip)
                    head_inc_op = arith_d.AddIOp(
                        lhs=head_val_op, rhs=const_one, ip=replace_ip
                    )
                    head_next_op = arith_d.RemUIOp(
                        lhs=head_inc_op, rhs=const_fifo_depth, ip=replace_ip
                    )
                # Use a while loop like the omp one, but suspend in it
                # Construct the while op, same for get and put
                suspend_while_op = scf_d.WhileOp(results_=[], inits=[], ip=replace_ip)
                assert isinstance(suspend_while_op.before, Region)
                assert isinstance(suspend_while_op.after, Region)
                before_block = Block.create_at_start(
                    parent=suspend_while_op.before, arg_types=[]
                )
                before_ip = InsertionPoint(before_block)
                after_block = Block.create_at_start(
                    parent=suspend_while_op.after, arg_types=[]
                )
                after_ip = InsertionPoint(after_block)
                multi_block_op = scf_d.ExecuteRegionOp([], ip=after_ip)
                scf_d.YieldOp(results_=[], ip=after_ip)
                block_0 = Block.create_at_start(multi_block_op.region, [])
                sub_suspend_block = Block.create_after(block_0)
                sub_cleanup_block = Block.create_after(sub_suspend_block)
                resume_block = Block.create_after(sub_cleanup_block)
                ip_0 = InsertionPoint(block_0)
                llvm_updated_save_op = llvm_d.CoroSaveOp(
                    res=llvm_token_type, handle=coro_handle, ip=ip_0
                )
                # llvm_updated_save_op = llvm_d.NoneTokenOp(ip=ip_0)
                llvm_suspend_op = llvm_d.CoroSuspendOp(
                    res=int8_type,
                    save=llvm_updated_save_op,
                    final=llvm_const_false,
                    ip=ip_0,
                )
                cf_d.SwitchOp(
                    flag=llvm_suspend_op,
                    defaultOperands=[],
                    defaultDestination=sub_suspend_block,
                    case_values=DenseIntElementsAttr.get(
                        [IntegerAttr.get(int8_type, 0), IntegerAttr.get(int8_type, 1)]
                    ),
                    caseOperands=[],
                    case_operand_segments=DenseI32ArrayAttr.get([0, 0]),
                    caseDestinations=[resume_block, sub_cleanup_block],
                    ip=ip_0,
                )
                scf_d.YieldOp(results_=[], ip=InsertionPoint(resume_block))
                llvm_d.UnreachableOp(ip=InsertionPoint(sub_cleanup_block))
                sub_suspend_ip = InsertionPoint(sub_suspend_block)
                # Debug
                if func_def_op.name.value == "producer_0":
                    str_const_addr_op = llvm_d.AddressOfOp(
                        res=llvm_ptr_type,
                        global_name=FlatSymbolRefAttr.get(".str_producer_block_debug"),
                        ip=sub_suspend_ip,
                    )
                else:
                    str_const_addr_op = llvm_d.AddressOfOp(
                        res=llvm_ptr_type,
                        global_name=FlatSymbolRefAttr.get(".str_consumer_block_debug"),
                        ip=sub_suspend_ip,
                    )
                str_addr_op = llvm_d.GEPOp(
                    res=llvm_ptr_type,
                    base=str_const_addr_op,
                    dynamicIndices=[],
                    rawConstantIndices=DenseI32ArrayAttr.get([0]),
                    elem_type=llvm_ptr_type,
                    ip=sub_suspend_ip,
                )
                llvm_d.CallOp(
                    result=int32_type,
                    callee_operands=[str_addr_op],
                    op_bundle_operands=[],
                    op_bundle_sizes=DenseI32ArrayAttr.get([]),
                    op_bundle_tags=None,
                    callee=FlatSymbolRefAttr.get("printf"),
                    var_callee_type=printf_func_type,
                    ip=sub_suspend_ip,
                )
                # Increment the promise value on suspend
                old_index_load_op = llvm_d.LoadOp(
                    res=int32_type, addr=promise_op, ip=sub_suspend_ip
                )
                new_index_op = llvm_d.AddOp(
                    lhs=old_index_load_op,
                    rhs=llvm_const_1_32,
                    overflowFlags=None,
                    ip=sub_suspend_ip,
                )
                new_index_store_op = llvm_d.StoreOp(
                    value=new_index_op, addr=promise_op, ip=sub_suspend_ip
                )
                llvm_sub_none_op = llvm_d.NoneTokenOp(ip=sub_suspend_ip)
                llvm_sub_end_op = llvm_d.CoroEndOp(
                    res=bool_type,
                    handle=coro_handle,
                    unwind=llvm_const_false,
                    retvals=llvm_sub_none_op,
                    ip=sub_suspend_ip,
                )  # This is actually a placeholder, can't use multiple coro.end ops
                cf_d.BranchOp([], resume_block, ip=sub_suspend_ip)
                # Calculate the condition in the "Before" Region
                # Different for get and put
                if isinstance(stream_access_op, allo_d.StreamPutOp):
                    head_val_op = memref_d.LoadOp(
                        memref=head_ptr, indices=[], ip=before_ip
                    )
                    cmp_op = arith_d.CmpIOp(
                        predicate=0, lhs=head_val_op, rhs=tail_next_op, ip=before_ip
                    )
                    # # Debug
                    # str_const_addr_op = llvm_d.AddressOfOp(
                    #     res=llvm_ptr_type,
                    #     global_name=FlatSymbolRefAttr.get(".str_producer_debug"),
                    #     ip=before_ip,
                    # )
                    # str_addr_op = llvm_d.GEPOp(
                    #     res=llvm_ptr_type,
                    #     base=str_const_addr_op,
                    #     dynamicIndices=[],
                    #     rawConstantIndices=DenseI32ArrayAttr.get([0]),
                    #     elem_type=llvm_ptr_type,
                    #     ip=before_ip,
                    # )
                    # llvm_d.CallOp(
                    #     result=int32_type,
                    #     callee_operands=[str_addr_op, head_val_op, tail_val_op],
                    #     op_bundle_operands=[],
                    #     op_bundle_sizes=DenseI32ArrayAttr.get([]),
                    #     op_bundle_tags=None,
                    #     callee=FlatSymbolRefAttr.get("printf"),
                    #     var_callee_type=printf_func_type,
                    #     ip=before_ip,
                    # )
                else:
                    assert isinstance(stream_access_op, allo_d.StreamGetOp)
                    tail_val_op = memref_d.LoadOp(
                        memref=tail_ptr, indices=[], ip=before_ip
                    )
                    cmp_op = arith_d.CmpIOp(
                        predicate=0, lhs=head_val_op, rhs=tail_val_op, ip=before_ip
                    )
                    # # Debug: print head and tail before awaiting
                    # str_const_addr_op = llvm_d.AddressOfOp(
                    #     res=llvm_ptr_type,
                    #     global_name=FlatSymbolRefAttr.get(".str_consumer_debug"),
                    #     ip=before_ip,
                    # )
                    # str_addr_op = llvm_d.GEPOp(
                    #     res=llvm_ptr_type,
                    #     base=str_const_addr_op,
                    #     dynamicIndices=[],
                    #     rawConstantIndices=DenseI32ArrayAttr.get([0]),
                    #     elem_type=llvm_ptr_type,
                    #     ip=before_ip,
                    # )
                    # llvm_d.CallOp(
                    #     result=int32_type,
                    #     callee_operands=[str_addr_op, head_val_op, tail_val_op],
                    #     op_bundle_operands=[],
                    #     op_bundle_sizes=DenseI32ArrayAttr.get([]),
                    #     op_bundle_tags=None,
                    #     callee=FlatSymbolRefAttr.get("printf"),
                    #     var_callee_type=printf_func_type,
                    #     ip=before_ip,
                    # )
                scf_d.ConditionOp(condition=cmp_op, args=[], ip=before_ip)
                if isinstance(stream_access_op, allo_d.StreamPutOp):
                    # Begin store
                    data = stream_access_op.data
                    assert isinstance(data, Value)
                    tail_index_op = index_d.CastUOp(
                        output=IndexType.get(module.context),
                        input=tail_val_op,
                        ip=replace_ip,
                    )
                    if isinstance(data.type, MemRefType):  # Vector
                        # Data is an `alloc` pointer and should be loaded first
                        element_type = data.type.element_type
                        if not isinstance(element_type, (IntegerType, FloatType)):
                            # May get StructType involved in the future
                            raise NotImplementedError()
                        rank = data.type.rank
                        assert rank > 0
                        for_ip = replace_ip
                        for_induction_vars = []
                        for_ips: list[InsertionPoint] = (
                            []
                        )  # Reserved to insert affine.yield ops later
                        for i in range(rank):
                            dim_size = data.type.get_dim_size(i)
                            for_loop_op = affine_d.AffineForOp(0, dim_size, ip=for_ip)
                            for_induction_vars.append(for_loop_op.induction_variable)
                            for_ip = InsertionPoint(for_loop_op.body)
                            for_ips.append(for_ip)
                        element_dim_map = AffineMap.get(
                            dim_count=rank,
                            symbol_count=0,
                            exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                            context=module.context,
                        )
                        element_load_op = affine_d.AffineLoadOp(
                            result=element_type,
                            memref=data,
                            indices=for_induction_vars,
                            map=AffineMapAttr.get(element_dim_map),
                            ip=for_ip,
                        )  # Fetch the element
                        memref_d.StoreOp(
                            value=element_load_op,
                            memref=fifo_ptr,
                            indices=[tail_index_op] + for_induction_vars,
                            ip=for_ip,
                        )  # Put the element to the stream
                        for ip in for_ips:
                            affine_d.AffineYieldOp([], ip=ip)
                    else:  # Scalar
                        memref_d.StoreOp(
                            value=data,
                            memref=fifo_ptr,
                            indices=[tail_index_op],
                            ip=replace_ip,
                        )
                    # End data store
                    # Atomic update tail to make the change visible
                    update_tail_op = memref_d.GenericAtomicRMWOp(
                        result=int32_type, memref=tail_ptr, indices=[], ip=replace_ip
                    )
                    update_tail_block = Block.create_at_start(
                        update_tail_op.atomic_body, [int32_type]
                    )
                    update_tail_ip = InsertionPoint(update_tail_block)
                    tail_inc_op = arith_d.AddIOp(
                        lhs=update_tail_block.arguments[0],
                        rhs=const_one,
                        ip=update_tail_ip,
                    )
                    tail_mod_size_op = arith_d.RemUIOp(
                        lhs=tail_inc_op, rhs=const_fifo_depth, ip=update_tail_ip
                    )
                    memref_d.AtomicYieldOp(result=tail_mod_size_op, ip=update_tail_ip)
                else:  # stream_get
                    assert isinstance(stream_access_op, allo_d.StreamGetOp)
                    # Begin load
                    orig_got_val = stream_access_op.res
                    assert isinstance(orig_got_val, OpResult)
                    head_index_op = index_d.CastUOp(
                        output=IndexType.get(module.context),
                        input=head_val_op,
                        ip=replace_ip,
                    )
                    if isinstance(orig_got_val.type, MemRefType):
                        element_type = orig_got_val.type.element_type
                        if not isinstance(element_type, (IntegerType, FloatType)):
                            raise NotImplementedError()
                        rank = orig_got_val.type.rank
                        assert rank > 0
                        # Create a memref for the loaded element
                        element_alloc_op = memref_d.AllocOp(
                            memref=orig_got_val.type,
                            dynamicSizes=[],
                            symbolOperands=[],
                            ip=replace_ip,
                        )
                        orig_got_val.replace_all_uses_with(element_alloc_op.result)
                        # Create the element load/store loop
                        for_ip = replace_ip
                        for_induction_vars = []
                        for_ips: list[InsertionPoint] = []
                        for i in range(rank):
                            for_loop_op = affine_d.AffineForOp(
                                0,
                                orig_got_val.type.get_dim_size(i),
                                ip=for_ip,
                            )
                            for_induction_vars.append(for_loop_op.induction_variable)
                            for_ip = InsertionPoint(for_loop_op.body)
                            for_ips.append(for_ip)
                        element_dim_map = AffineMap.get(
                            dim_count=rank,
                            symbol_count=0,
                            exprs=[AffineExpr.get_dim(i) for i in range(rank)],
                            context=module.context,
                        )
                        element_load_op = memref_d.LoadOp(
                            memref=fifo_ptr,
                            indices=[head_index_op] + for_induction_vars,
                            ip=for_ip,  # The innermost Loop body
                        )
                        affine_d.AffineStoreOp(
                            value=element_load_op,
                            memref=element_alloc_op,
                            indices=for_induction_vars,
                            map=AffineMapAttr.get(element_dim_map),
                            ip=for_ip,
                        )
                        for ip in for_ips:
                            affine_d.AffineYieldOp([], ip=ip)
                    else:  # Scalar
                        new_get_op = memref_d.LoadOp(
                            memref=fifo_ptr, indices=[head_index_op], ip=replace_ip
                        )
                        orig_got_val.replace_all_uses_with(new_get_op.result)
                    # End load (not visible yet)
                    # Update head to make update visible
                    update_head_op = memref_d.GenericAtomicRMWOp(
                        result=int32_type, memref=head_ptr, indices=[], ip=replace_ip
                    )
                    update_head_block = Block.create_at_start(
                        parent=update_head_op.atomic_body, arg_types=[int32_type]
                    )
                    update_head_ip = InsertionPoint(update_head_block)
                    head_inc_1_op = arith_d.AddIOp(
                        lhs=update_head_block.arguments[0],
                        rhs=const_one,
                        ip=update_head_ip,
                    )
                    head_mod_size_op = arith_d.RemUIOp(
                        lhs=head_inc_1_op, rhs=const_fifo_depth, ip=update_head_ip
                    )
                    memref_d.AtomicYieldOp(result=head_mod_size_op, ip=update_head_ip)
                stream_access_op.operation.erase()
        for op in stream_construct_ops.values():
            op.operation.erase()

        assert len(pe_call_define_ops) > 0
        call_ip = InsertionPoint(beforeOperation=list(pe_call_define_ops.keys())[0])
        dummy_op = arith_d.ConstantOp(int32_type, 1, ip=call_ip)
        call_ip = InsertionPoint(beforeOperation=dummy_op)
        handles = []
        # Create new call operations since their return types have changed
        # Init the coroutines with the first calls
        for call_op in pe_call_define_ops:
            new_call_op = func_d.CallOp(
                [llvm_ptr_type],  # Result
                call_op.callee,  # Callee
                call_op.operands_,  # Arguments
                ip=call_ip,
            )
            call_op.operation.erase()
            handles.append(new_call_op)

        # "Schedule" the tasks
        num_pes = len(handles)
        llvm_const_false = llvm_d.ConstantOp(
            res=bool_type, value=IntegerAttr.get(bool_type, 0), ip=call_ip
        )
        llvm_const_minus1_32 = llvm_d.ConstantOp(
            res=int32_type, value=IntegerAttr.get(int32_type, -1), ip=call_ip
        )
        llvm_const_true = llvm_d.ConstantOp(
            res=bool_type, value=IntegerAttr.get(bool_type, 1), ip=call_ip
        )
        return_addr_op = llvm_d.AddressOfOp(
            llvm_ptr_type, FlatSymbolRefAttr.get("__resume"), ip=call_ip
        )
        # For every task, start it for 1 time unconditionally
        for handle in handles:
            llvm_d.CallOp(
                result=None,
                callee_operands=[handle, return_addr_op],
                op_bundle_operands=[],
                op_bundle_sizes=[],
                op_bundle_tags=[],
                callee=FlatSymbolRefAttr.get("mlirAsyncRuntimeExecute"),
                ip=call_ip,
            )
        # Resume the tasks with the while loop logic
        promise_tracks_vals = []
        if not const_0_defined:
            const_zero = arith_d.ConstantOp(int32_type, 0, ip=call_ip)
        for i in range(num_pes):
            promise_track = memref_d.AllocOp(memref_scalar_int_type, [], [], ip=call_ip)
            memref_d.StoreOp(
                value=const_zero,
                memref=promise_track,
                indices=[],
                ip=call_ip,
            )
            promise_tracks_vals.append(promise_track)
        coro_call_loop_op = scf_d.WhileOp(results_=[], inits=[], ip=call_ip)  # Do-While
        assert isinstance(coro_call_loop_op.before, Region)
        assert isinstance(coro_call_loop_op.after, Region)
        before_block = Block.create_at_start(coro_call_loop_op.before, [])
        before_ip = InsertionPoint(before_block)
        after_block = Block.create_at_start(coro_call_loop_op.after, [])
        after_ip = InsertionPoint(after_block)
        # Resume each task in the before block
        for i in range(num_pes):
            handle = handles[i]
            tracked_promise_val = promise_tracks_vals[i]
            prev_promise_load_op = memref_d.LoadOp(
                memref=tracked_promise_val, indices=[], ip=before_ip
            )
            cur_promise_op = llvm_d.CoroPromiseOp(
                res=llvm_ptr_type,
                handle=handle,
                from_=llvm_const_false,
                align=const_zero,
                ip=before_ip,
            )
            cur_promise_load_op = llvm_d.LoadOp(
                res=int32_type, addr=cur_promise_op, ip=before_ip
            )
            this_not_done_cond = arith_d.CmpIOp(
                predicate=1,  # ne
                lhs=cur_promise_load_op,
                rhs=llvm_const_minus1_32,
                ip=before_ip,
            )
            if_not_done_op = scf_d.IfOp(
                cond=this_not_done_cond, results_=[], ip=before_ip
            )
            if_not_done_ip = InsertionPoint(if_not_done_op.then_block)
            now_suspend_cond = arith_d.CmpIOp(
                predicate=2,  # slt
                lhs=prev_promise_load_op,
                rhs=cur_promise_load_op,
                ip=if_not_done_ip,
            )  # if the promise value become larger, the coro is suspended
            if_now_suspend_op = scf_d.IfOp(
                cond=now_suspend_cond, results_=[], ip=if_not_done_ip
            )  # if not done & suspended, resume the coroutine
            if_now_suspend_ip = InsertionPoint(if_now_suspend_op.then_block)
            # Update the tracked promise value
            memref_d.StoreOp(
                value=cur_promise_load_op,
                memref=tracked_promise_val,
                indices=[],
                ip=if_now_suspend_ip,
            )
            llvm_d.CallOp(
                result=None,
                callee_operands=[handle, return_addr_op],
                op_bundle_operands=[],
                op_bundle_sizes=[],
                op_bundle_tags=[],
                callee=FlatSymbolRefAttr.get("mlirAsyncRuntimeExecute"),
                ip=if_now_suspend_ip,
            )
            scf_d.YieldOp([], ip=if_now_suspend_ip)
            scf_d.YieldOp([], ip=if_not_done_ip)
            # Update the state of coroutines
            if i == 0:
                has_coro_not_done_op = this_not_done_cond
            else:
                updated_not_done_op = arith_d.OrIOp(
                    lhs=has_coro_not_done_op, rhs=this_not_done_cond, ip=before_ip
                )
                has_coro_not_done_op = updated_not_done_op
        scf_d.ConditionOp(  # Proceed if not all coroutines have finished
            condition=has_coro_not_done_op, args=[], ip=before_ip
        )
        scf_d.YieldOp([], ip=after_ip)
        dummy_op.operation.erase()


def recursive_find_blocks_containing_op(
    top_op: Operation, target_op_name: str, res: list
):
    for region in top_op.regions:
        for block in region.blocks:
            for op in block:
                if op.name == target_op_name:
                    res.append(block)
                recursive_find_blocks_containing_op(op, target_op_name, res)


# Remove the placeholder coro.end op and set correct fallthrough target
def clean_coro_end_ops(module: Module):
    with module.context, Location.unknown():
        coros = []
        for op in module.body:
            if isinstance(op, llvm_d.LLVMFuncOp):
                assert isinstance(op, OpView)
                if op.attributes.__contains__("passthrough"):
                    coros.append(op)
        for coro in coros:
            assert isinstance(coro, llvm_d.LLVMFuncOp)
            coro_end_blocks = []
            recursive_find_blocks_containing_op(
                coro, "llvm.intr.coro.end", coro_end_blocks
            )
            # Only the last block containing coro.end should be preserved
            actual = coro_end_blocks[-1]
            assert isinstance(actual, Block)
            dummies = coro_end_blocks[:-1]
            for dummy in dummies:
                assert isinstance(dummy, Block)
                remove_ops = []
                for op in dummy.operations:
                    if (
                        op.name == "llvm.mlir.none"
                        or op.name == "llvm.intr.coro.end"
                        or op.name == "llvm.br"
                    ):
                        remove_ops.append(op)
                for op in reversed(remove_ops):
                    op.operation.erase()
                ip = InsertionPoint(dummy)
                llvm_d.BrOp([], dest=actual, ip=ip)


class LLVMCoroModule(LLVMModule):
    def __init__(self, mod: Module, top_func_name: str, ext_libs=None):
        with Context() as ctx:
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            self.top_func_name = top_func_name
            func = find_func_in_module(self.module, top_func_name)
            ext_libs = [] if ext_libs is None else ext_libs
            # Get input/output types
            self.in_types, self.out_types = get_func_inputs_outputs(func)
            self.module = decompose_library_function(self.module)

            build_coroutine_dataflow_simulator(self.module, self.top_func_name)
            # Attach necessary attributes
            func = find_func_in_module(self.module, top_func_name)
            if func is None:
                raise RuntimeError(
                    "No top-level function found in the built MLIR module"
                )
            func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            func.attributes["top"] = UnitAttr.get()

            # Start lowering
            # Lower linalg for AIE
            pm = PassManager.parse(
                "builtin.module("
                "one-shot-bufferize,"
                "expand-strided-metadata,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)
            # Lower StructType
            allo_d.lower_composite_type(self.module)
            # print(self.module)
            pm = PassManager.parse(
                "builtin.module("
                "lower-affine,"
                "convert-scf-to-cf,"
                # "convert-async-to-llvm,"
                "convert-func-to-llvm,"
                "convert-index-to-llvm,"
                "convert-cf-to-llvm,"
                "finalize-memref-to-llvm,"
                "canonicalize"
                ")"
            )
            pm.run(self.module.operation)
            clean_coro_end_ops(self.module)

            assert os.getenv("LLVM_BUILD_DIR") is not None, "LLVM_BUILD_DIR is not set"
            shared_libs = [
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_runner_utils.so"
                ),
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_c_runner_utils.so"
                ),
                os.path.join(
                    os.getenv("LLVM_BUILD_DIR"), "lib", "libmlir_async_runtime.so"
                ),
            ]
            shared_libs += [lib.compile_shared_lib() for lib in ext_libs]
            self.execution_engine = ExecutionEngine(
                self.module, opt_level=2, shared_libs=shared_libs
            )
