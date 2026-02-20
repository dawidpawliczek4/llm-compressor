# Notatki z tworzenia algorytmu kompresji z uzyciem modeli jezykowych


## 1. LSTM i początek

Na poczatku stworzylem najprostszy algorytm, prosty model lstm

Baseline Results (po 10 epoch):
Compression Speed: 25121.45 B/s
Decompression Speed: 25530.53 B/s
Compression Ratio: 1.58x
BPC: 5.07

train time - epoch 2:30min


## 2. GRU i lepsze parametry

znacząco lepszy wynik juz po jednym epochu-- kosztem szybkości

na train small:
Baseline Results (po 1 epoch):
Compression Speed: 5901.51 B/s
Decompression Speed: 5840.39 B/s
Compression Ratio: 3.96x
BPC: 2.02

train time - epoch 10 minut

na test small:
Baseline Results:
Compression Ratio: 2.32x
BPC: 3.44

na test all_canterbury:
Baseline Results:
Compression Speed: 5931.78 B/s
Decompression Speed: 5840.65 B/s
Compression Ratio: 1.84x
BPC: 4.35

czas kompresji jest tragiczny:
Model Parameters: 4,302,208
=== COMPRESSION ===
Compressing 18521760 bytes...
Encoding: 100%|██████████| 18521760/18521760 [52:02<00:00, 5931.80it/s] 
Compression Ratio: 1.84x
BPC: 4.35
=== DECOMPRESSION ===
Decompressing 18521760 bytes...
Decoding: 100%|██████████| 18521760/18521760 [52:51<00:00, 5840.74it/s] 
SUCCESS: Integrity verified!



ALE NA MALYCH DANYCH (250 linijek z ksiazki z dancyh testowych) mamy dobre resultaty:
/Users/dawidpawliczek/Developer/llm-compressor/.venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Using device: CPU
Model Parameters: 4,302,208
=== COMPRESSION ===
Compressing 10846 bytes...
Encoding: 100%|██████████| 10846/10846 [00:01<00:00, 5800.55it/s]

Compression time profile:
  model_infer        1.7539s  ( 93.70%)
  state_update       0.0287s  (  1.54%)
  arith_encode       0.0171s  (  0.91%)
  write_file         0.0010s  (  0.05%)
  read_bytes         0.0001s  (  0.00%)
  other              0.0710s  (  3.79%)
Compression Ratio: 2.89x
BPC: 2.77
=== DECOMPRESSION ===
Decompressing 10846 bytes...
Decoding: 100%|██████████| 10846/10846 [00:01<00:00, 5840.72it/s]

Decompression time profile:
  model_infer        1.7456s  ( 93.94%)
  state_update       0.0278s  (  1.50%)
  arith_decode       0.0160s  (  0.86%)
  write_file         0.0004s  (  0.02%)
  read_file          0.0001s  (  0.01%)
  other              0.0683s  (  3.68%)
SUCCESS: Integrity verified!
Baseline Results:
Compression Speed: 5794.42 B/s
Decompression Speed: 5836.60 B/s
Compression Ratio: 2.89x
BPC: 2.77



DLA apple silicon MPU

Model Parameters: 4,302,208
=== COMPRESSION ===
Compressing 10846 bytes...
Encoding: 100%|██████████| 10846/10846 [00:07<00:00, 1419.75it/s]

Compression time profile:
  model_infer        6.0512s  ( 79.20%)
  state_update       1.4888s  ( 19.48%)
  arith_encode       0.0150s  (  0.20%)
  write_file         0.0004s  (  0.01%)
  read_bytes         0.0002s  (  0.00%)
  other              0.0850s  (  1.11%)
Compression Ratio: 2.89x
BPC: 2.77
=== DECOMPRESSION ===
Decompressing 10846 bytes...
Decoding: 100%|██████████| 10846/10846 [00:07<00:00, 1432.57it/s]

Decompression time profile:
  model_infer        6.0064s  ( 79.32%)
  state_update       1.4609s  ( 19.29%)
  arith_decode       0.0150s  (  0.20%)
  write_file         0.0003s  (  0.00%)
  read_file          0.0001s  (  0.00%)
  other              0.0898s  (  1.19%)
SUCCESS: Integrity verified!



DLA CUDA:
Model Parameters: 4,302,208
=== COMPRESSION ===
Compressing 10846 bytes...
Encoding: 100%
 10846/10846 [00:06<00:00, 1916.24it/s]

Compression time profile:
  model_infer        5.3898s  ( 83.16%)
  state_update       0.5794s  (  8.94%)
  arith_encode       0.0915s  (  1.41%)
  write_file         0.0003s  (  0.00%)
  read_bytes         0.0003s  (  0.00%)
  other              0.4196s  (  6.47%)
Compression Ratio: 2.89x
BPC: 2.77
=== DECOMPRESSION ===
Decompressing 10846 bytes...
Decoding: 100%
 10846/10846 [00:05<00:00, 1888.14it/s]

Decompression time profile:
  model_infer        4.7111s  ( 82.64%)
  state_update       0.5302s  (  9.30%)
  arith_decode       0.0799s  (  1.40%)
  write_file         0.0004s  (  0.01%)
  read_file          0.0001s  (  0.00%)
  other              0.3793s  (  6.65%)
SUCCESS: Integrity verified!


## 3. GPT-2

Test file: ../data/canterbury_small.bin
Loading gpt2...
Loading weights: 100%|██████████| 148/148 [00:00<00:00, 2716.73it/s, Materializing param=transformer.wte.weight]             
GPT2LMHeadModel LOAD REPORT from: gpt2
Key                  | Status     |  | 
---------------------+------------+--+-
h.{0...11}.attn.bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Token indices sequence length is longer than the specified maximum sequence length for this model (3064 > 1024). Running this sequence through the model will result in indexing errors
Loaded: 124,439,808 parameters on cpu

=== COMPRESSION ===
Original: 10,846 bytes → 3,064 tokens
Compressing: 100%|██████████| 3064/3064 [00:25<00:00, 122.24it/s]

Compression time profile:
  model_infer       24.7669s  ( 98.78%)
  arith_encode       0.2077s  (  0.83%)
  tokenize           0.0041s  (  0.02%)
  write_file         0.0005s  (  0.00%)
  read_bytes         0.0001s  (  0.00%)
  other              0.0931s  (  0.37%)
Ratio: 4.75x | BPC: 1.68

=== DECOMPRESSION ===
Decompressing: 3,064 tokens → 10,846 bytes
Decompressing: 100%|██████████| 3064/3064 [00:25<00:00, 119.55it/s]

Decompression time profile:
  model_infer       25.3027s  ( 98.72%)
  arith_decode       0.2215s  (  0.86%)
  detokenize         0.0010s  (  0.00%)
  write_file         0.0004s  (  0.00%)
  read_file          0.0000s  (  0.00%)
  other              0.1056s  (  0.41%)
Speed: 423.16 B/s

=== VERIFICATION ===
✅ SUCCESS: Perfect match!

======================================================================
                        COMPRESSION COMPARISON                        
======================================================================

Original: 10,846 bytes

Method               Size         Ratio      BPC        Speed (B/s)
----------------------------------------------------------------------
GPT-2                2,284        4.75       1.68       432.59
ZIP (level 9)        5,189        2.09       3.83       17902959.93

📊 GPT-2 vs ZIP:
...
Compression Speed: 432.59 B/s
Decompression Speed: 423.16 B/s
Compression Ratio: 4.75x
BPC: 1.68



----- distilgpt2

== COMPRESSION ===
Original: 10,846 bytes → 3,064 tokens
Compressing: 100%|██████████| 3064/3064 [00:14<00:00, 210.98it/s]

Compression time profile:
  model_infer       14.2553s  ( 98.08%)
  arith_encode       0.2103s  (  1.45%)
  tokenize           0.0090s  (  0.06%)
  read_bytes         0.0005s  (  0.00%)
  write_file         0.0004s  (  0.00%)
  other              0.0588s  (  0.40%)
Ratio: 4.20x | BPC: 1.90

=== DECOMPRESSION ===
Decompressing: 3,064 tokens → 10,846 bytes
Decompressing: 100%|██████████| 3064/3064 [00:13<00:00, 227.10it/s]

Decompression time profile:
  model_infer       13.2378s  ( 98.11%)
  arith_decode       0.2019s  (  1.50%)
  detokenize         0.0005s  (  0.00%)
  write_file         0.0004s  (  0.00%)
  read_file          0.0000s  (  0.00%)
  other              0.0528s  (  0.39%)
Speed: 803.80 B/s

📊 DistilGPT-2 vs ZIP:
...
Compression Speed: 746.24 B/s
Decompression Speed: 803.80 B/s
Compression Ratio: 4.20x
BPC: 1.90


## 4. Podsumowanie tabelaryczne i wnioski

### 4.1 Porównanie modeli (na podstawie notatek)

| Model | Zbiór | Compression (B/s) | Decompression (B/s) | Ratio (x) | BPC |
|---|---|---:|---:|---:|---:|
| LSTM (10 ep) | small | 25121.45 | 25530.53 | 1.58 | 5.07 |
| GRU (1 ep) | small (train) | 5901.51 | 5840.39 | 3.96 | 2.02 |
| GRU | all_canterbury | 5931.78 | 5840.65 | 1.84 | 4.35 |
| GRU | canterbury 250 lines | 5794.42 | 5836.60 | 2.89 | 2.77 |
| GPT-2 | canterbury_small.bin | 432.59 | 423.16 | 4.75 | 1.68 |
| DistilGPT-2 | canterbury_small.bin | 746.24 | 803.80 | 4.20 | 1.90 |

### 4.2 GRU — porównanie urządzeń (ten sam wycinek 10,846 B)

| Device | Compression (B/s) | Decompression (B/s) | model_infer % (comp) | model_infer % (decomp) |
|---|---:|---:|---:|---:|
| CPU | 5800.55 | 5840.72 | 93.70% | 93.94% |
| Apple MPS | 1419.75 | 1432.57 | 79.20% | 79.32% |
| CUDA T4 | 1916.24 | 1888.14 | 83.16% | 82.64% |

### 4.3 Wnioski końcowe

- GPT-2 daje najlepszą jakość kompresji (najniższy BPC), ale jest wolny.
- DistilGPT-2 jest dobrym kompromisem: znacznie szybszy od GPT-2 przy umiarkowanej utracie jakości.
- GRU bywa konkurencyjny na małych, jednorodnych danych, ale gorzej generalizuje na pełnym, mieszanym Canterbury.
- W modelach neuronowych główny bottleneck to inferencja modelu (`model_infer` zwykle >80% całego czasu).
- Klasyczne kompresory (ZIP) nadal wygrywają szybkością o rzędy wielkości.