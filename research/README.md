# LLM Compressor CLI

CLI jest dostępne jako komenda `llmc` i używa modelu `distilgpt2` + arithmetic coding.

## Instalacja zależności

W katalogu `research`:

```bash
uv sync
```

albo klasycznie:

```bash
pip install -e .
```

## Instalacja komendy `llmc` (globalnie)

Żeby komenda `llmc` działała z dowolnego katalogu, zainstaluj tool globalnie:

```bash
uv tool install --from . llm-compressor
```

Przy kolejnych zmianach kodu CLI przeinstaluj:

```bash
uv tool install --reinstall --from . llm-compressor
```

Sprawdzenie, czy komenda jest dostępna:

```bash
llmc --help
```

Jeśli komenda nie jest widoczna, dodaj `~/.local/bin` do `PATH`.

## Użycie

Kompresja:

```bash
llmc compress ./plik.txt ./out.bin
```

Dekompresja:

```bash
llmc decompress ./out.bin ./plik_odtworzony.txt
```

Opcjonalne flagi:

- `--model` (domyślnie: `distilgpt2`)
- `--context-size` (domyślnie: `1024`)
- `--stride` (domyślnie: `512`)
