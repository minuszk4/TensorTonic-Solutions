import numpy as np

def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # # Write code here
    # if chunk_size <= 0:
    #     raise ValueError("chunk_size must be positive.")
    # if overlap < 0:
    #     raise ValueError("overlap must be non-negative.")
    step = chunk_size - overlap
    # if step <= 0:
        # raise ValueError("overlap must be less than chunk_size.")
    chunks = []
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) < chunk_size and len(chunks) > 0:
            break
        chunks.append(chunk)
    return chunks