"""
Debug script to find the mismatch between vectorized compression and decompression.

FINDING: LSTM batched computation gives slightly different floating-point results
than symbol-by-symbol computation. This tiny difference (10^-10) is enough to
completely break entropy coding.

SOLUTION: For entropy coding, compression MUST use the EXACT same computational
path as decompression. The "vectorization" speedup CANNOT come from batching the
model forward pass.
"""
import torch
import torch.nn as nn
import numpy as np
import constriction

HIDDEN_SIZE = 128
DEVICE = torch.device("cpu")


class Compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(257, 32)
        self.lstm = nn.LSTM(32, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 256)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def get_probs(model, symbol, hidden):
    """Get probability distribution for next symbol. Used by both compress and decompress."""
    with torch.no_grad():
        x = torch.tensor([[symbol]], dtype=torch.long, device=DEVICE)
        logits, hidden = model(x, hidden)
        probs = torch.softmax(logits[0, 0], dim=0).cpu().numpy().astype(np.float32)
    return probs, hidden


def compress(model, data):
    """Compress data - symbol by symbol. Uses EXACT same logic as decompress."""
    model.eval()
    encoder = constriction.stream.queue.RangeEncoder()
    
    hidden = None
    curr_symbol = 256  # START
    
    for symbol in data:
        probs, hidden = get_probs(model, curr_symbol, hidden)
        dist = constriction.stream.model.Categorical(probs, perfect=False)
        encoder.encode(int(symbol), dist)
        curr_symbol = int(symbol)
    
    return encoder.get_compressed()


def decompress(model, compressed_bits, length):
    """Decompress data - symbol by symbol. Uses EXACT same logic as compress."""
    model.eval()
    decoder = constriction.stream.queue.RangeDecoder(compressed_bits)
    
    decoded = []
    hidden = None
    curr_symbol = 256  # START
    
    with torch.no_grad():
        for _ in range(length):
            probs, hidden = get_probs(model, curr_symbol, hidden)
            dist = constriction.stream.model.Categorical(probs, perfect=False)
            symbol = decoder.decode(dist)
            decoded.append(symbol)
            curr_symbol = symbol
    
    return bytes(decoded)


def compress_broken_vectorized(model, data, chunk_size=1000):
    """
    BROKEN: Attempt at vectorization by computing logits in chunks.
    
    This WON'T work because LSTM batched computation gives slightly different 
    floating-point results than symbol-by-symbol computation.
    
    Kept here to demonstrate WHY simple vectorization fails.
    """
    model.eval()
    encoder = constriction.stream.queue.RangeEncoder()
    
    length = len(data)
    last_symbol = 256
    hidden = None
    
    with torch.no_grad():
        for i in range(0, length, chunk_size):
            chunk_target = data[i : i + chunk_size]
            chunk_len = len(chunk_target)
            
            input_seq = np.empty(chunk_len, dtype=np.int64)
            input_seq[0] = last_symbol
            if chunk_len > 1:
                input_seq[1:] = chunk_target[:-1]
            
            input_tensor = torch.from_numpy(input_seq).unsqueeze(0).to(DEVICE)
            logits, hidden = model(input_tensor, hidden)
            
            for j in range(chunk_len):
                symbol = chunk_target[j]
                probs = torch.softmax(logits[0, j], dim=0).cpu().numpy().astype(np.float32)
                dist = constriction.stream.model.Categorical(probs, perfect=False)
                encoder.encode(int(symbol), dist)
            
            last_symbol = int(chunk_target[-1])
    
    return encoder.get_compressed()


def test_compression():
    """Test that compress and decompress are inverses."""
    torch.manual_seed(42)
    model = Compressor().to(DEVICE)
    
    test_data = b"Hello, World! This is a test of the compression system."
    data_array = np.frombuffer(test_data, dtype=np.uint8)
    
    print(f"Test data: {len(test_data)} bytes")
    print(f"Content: {test_data}")
    
    # Test correct implementation
    print("\n=== Testing CORRECT (symbol-by-symbol) compression ===")
    compressed = compress(model, data_array)
    print(f"Compressed size: {len(compressed) * 4} bytes")
    
    decompressed = decompress(model, compressed, len(test_data))
    correct_ok = decompressed == test_data
    print(f"Roundtrip: {'✓ SUCCESS' if correct_ok else '✗ FAILED'}")
    
    # Test broken vectorized implementation  
    print("\n=== Testing BROKEN (vectorized) compression ===")
    compressed_broken = compress_broken_vectorized(model, data_array, chunk_size=20)
    print(f"Compressed size: {len(compressed_broken) * 4} bytes")
    
    try:
        decompressed_broken = decompress(model, compressed_broken, len(test_data))
        broken_ok = decompressed_broken == test_data
        print(f"Roundtrip: {'✓ SUCCESS' if broken_ok else '✗ FAILED'}")
        
        if not broken_ok:
            for i, (orig, dec) in enumerate(zip(test_data, decompressed_broken)):
                if orig != dec:
                    print(f"First mismatch at position {i}: expected {orig} ({chr(orig)}), got {dec}")
                    break
    except Exception as e:
        print(f"Decompression failed with error: {e}")
    
    # Show WHY it fails - probability differences
    print("\n=== Root Cause Analysis ===")
    print("Comparing probabilities between symbol-by-symbol and batched computation:")
    
    # Symbol-by-symbol
    hidden1 = None
    probs1, hidden1 = get_probs(model, 256, hidden1)  # P(next | START)
    
    # Batched (single element for comparison)
    with torch.no_grad():
        x = torch.tensor([[256]], dtype=torch.long, device=DEVICE)
        logits2, _ = model(x, None)
        probs2 = torch.softmax(logits2[0, 0], dim=0).cpu().numpy().astype(np.float32)
    
    print(f"For first symbol (START):")
    print(f"  Symbol-by-symbol: P[72]={probs1[72]:.10f}")
    print(f"  Batched:          P[72]={probs2[72]:.10f}")
    print(f"  Exact match: {np.array_equal(probs1, probs2)}")
    print(f"  Max difference: {np.abs(probs1 - probs2).max():.2e}")
    
    # Now test with longer batch
    input_batch = torch.tensor([[256, 72, 101]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits_batch, _ = model(input_batch, None)
        probs_batch_0 = torch.softmax(logits_batch[0, 0], dim=0).cpu().numpy().astype(np.float32)
    
    print(f"\nWhen input is batched [256, 72, 101]:")
    print(f"  Batched P[72] at position 0: {probs_batch_0[72]:.10f}")
    print(f"  Symbol-by-symbol P[72]:      {probs1[72]:.10f}")
    print(f"  Exact match: {np.array_equal(probs1, probs_batch_0)}")
    print(f"  Max difference: {np.abs(probs1 - probs_batch_0).max():.2e}")


if __name__ == "__main__":
    test_compression()
