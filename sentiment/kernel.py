import triton.language as tl
import triton


'''

Hidden states evolution GPU calc. kernel for LTC model, can be solved using DormanPrince 5 or RungeKutta 4th Order

'''

@triton.jit
def LiKernel(
    h_ptr,
    input_ptr,
    output_ptr,
    weight_rec_ptr,
    weight_in_ptr,
    weight_gate_ptr,
    log_tau_ptr,
    batch_size,
    hidden_size,
    input_size,
    stride_h,
    stride_in,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (hidden_size // BLOCK_SIZE)
    start_idx = (pid % (hidden_size // BLOCK_SIZE)) * BLOCK_SIZE
    h_batch_ptr = h_ptr + batch_idx * stride_h
    input_batch_ptr = input_ptr + batch_idx * stride_in
    output_batch_ptr = output_ptr + batch_idx * stride_out
    offsets = tl.arange(0, BLOCK_SIZE)
    h = tl.load(h_batch_ptr + start_idx + offsets)
    log_tau = tl.load(log_tau_ptr + start_idx + offsets)
    exp_log_tau = tl.exp(log_tau)
    tau = tl.log(1.0 + exp_log_tau) + 0.1   
    gate_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    recurrent_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    for j in range(0, hidden_size, BLOCK_SIZE):
        weight_gate_block_ptr = weight_gate_ptr + (start_idx + offsets)[:, None] * hidden_size + j + offsets[None, :]
        weight_rec_block_ptr = weight_rec_ptr + (start_idx + offsets)[:, None] * hidden_size + j + offsets[None, :]
        h_block_ptr = h_batch_ptr + j + offsets
        weight_gate_block = tl.load(weight_gate_block_ptr)
        weight_rec_block = tl.load(weight_rec_block_ptr)
        h_block = tl.load(h_block_ptr)
        gate_product = weight_gate_block * h_block[:, None]
        recurrent_product = weight_rec_block * h_block[:, None]
        gate_sum += tl.sum(gate_product, axis=0)
        recurrent_sum += tl.sum(recurrent_product, axis=0)
    gate_exp = tl.exp(-gate_sum)
    gate_term = 1.0 / (1.0 + gate_exp)
    recurrent_inner = 2.0 * recurrent_sum
    recurrent_exp = tl.exp(-recurrent_inner)
    recurrent_sigmoid = 1.0 / (1.0 + recurrent_exp)
    recurrent_term = 2.0 * recurrent_sigmoid - 1.0
    recurrent_term = gate_term * recurrent_term
    input_term = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    for i in range(0, input_size, BLOCK_SIZE):
        weight_in_block_ptr = weight_in_ptr + (start_idx + offsets)[:, None] * input_size + i + offsets[None, :]
        x_block_ptr = input_batch_ptr + i + offsets
        weight_in_block = tl.load(weight_in_block_ptr)
        x_block = tl.load(x_block_ptr)
        input_term += tl.sum(weight_in_block * x_block[:, None], axis=0)
    numerator = (-h) + recurrent_term + input_term
    dhdt = numerator / tau
    tl.store(output_batch_ptr + start_idx + offsets, dhdt)
