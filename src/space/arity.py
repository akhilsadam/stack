import torch

# @torch.jit.script
def get_tree_coords(arities: torch.Tensor):
    """
    JIT-compiled sequential stack parser for RPN tree coordinates.
    Compiles to C++ to bypass Python interpreter overhead.
    """
    batch_size, seq_len = arities.shape
    depths = torch.zeros_like(arities)
    horizontals = torch.zeros_like(arities)
    
    for b in range(batch_size):
        # Pre-allocate memory to avoid dynamic list resizing
        stack = torch.zeros(seq_len, dtype=torch.long)
        depth_counts = torch.zeros(seq_len, dtype=torch.long)
        stack_ptr = 0
        
        for i in range(seq_len):
            arity = arities[b, i]
            
            if arity == 0:
                curr_depth = 0
            else:
                # Pop 'arity' items and find the max depth
                max_d = -1
                for _ in range(arity):
                    stack_ptr -= 1
                    d = stack[stack_ptr].item()
                    if d > max_d:
                        max_d = int(d)
                curr_depth = max_d + 1
                
            # Assign horizontal index and increment counter
            curr_horiz = int(depth_counts[curr_depth].item())
            depth_counts[curr_depth] = curr_horiz + 1
            
            # Push current depth back to stack
            stack[stack_ptr] = curr_depth
            stack_ptr += 1
            
            # Write to output
            depths[b, i] = curr_depth
            horizontals[b, i] = curr_horiz
            
    return depths, horizontals