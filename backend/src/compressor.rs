use anyhow::{Context, Result, anyhow};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bigcode::{Config, GPTBigCode};
use constriction::stream::{
    Decode, Encode,
    model::DefaultLazyContiguousCategoricalEntropyModel,
    queue::{DefaultRangeDecoder, DefaultRangeEncoder},
};
use serde::Deserialize;
use std::collections::HashSet;
use std::io::Cursor;
use tokenizers::Tokenizer;

const CONTEXT_SIZE: usize = 512;
const STRIDE: usize = 256;
const MODEL_ID: &str = "bigcode/tiny_starcoder_py";

#[derive(Debug, Deserialize, Clone)]
struct BigCodeConfigFile {
    vocab_size: usize,
    n_positions: usize,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    layer_norm_epsilon: f64,
    n_inner: Option<usize>,
    #[serde(default)]
    multi_query: bool,
    #[serde(default = "default_true")]
    use_cache: bool,
}

#[derive(Debug, Deserialize)]
struct WeightsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

fn default_true() -> bool {
    true
}

pub struct LLMCompressor {
    model: GPTBigCode,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
    cfg_raw: BigCodeConfigFile,
    weight_files: Vec<std::path::PathBuf>,
}

impl LLMCompressor {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(MODEL_ID.to_string());

        let config_path = repo.get("config.json")?;
        let config_str = std::fs::read_to_string(config_path)?;
        let raw_cfg: BigCodeConfigFile = serde_json::from_str(&config_str)?;
        let cfg = Config {
            vocab_size: raw_cfg.vocab_size,
            max_position_embeddings: raw_cfg.n_positions,
            num_hidden_layers: raw_cfg.n_layer,
            hidden_size: raw_cfg.n_embd,
            layer_norm_epsilon: raw_cfg.layer_norm_epsilon,
            n_inner: raw_cfg.n_inner,
            num_attention_heads: raw_cfg.n_head,
            multi_query: raw_cfg.multi_query,
            // KLUCZOWE: cache on (jak w notebooku)
            use_cache: true,
        };

        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e.to_string()))?;
        let eos_token_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);

        let weight_files = Self::resolve_weight_files(&repo)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, &device)? };
        let model = GPTBigCode::load(vb, cfg)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token_id,
            cfg_raw: raw_cfg,
            weight_files,
        })
    }

    fn build_config_from_raw(raw_cfg: &BigCodeConfigFile) -> Config {
        Config {
            vocab_size: raw_cfg.vocab_size,
            max_position_embeddings: raw_cfg.n_positions,
            num_hidden_layers: raw_cfg.n_layer,
            hidden_size: raw_cfg.n_embd,
            layer_norm_epsilon: raw_cfg.layer_norm_epsilon,
            n_inner: raw_cfg.n_inner,
            num_attention_heads: raw_cfg.n_head,
            multi_query: raw_cfg.multi_query,
            use_cache: true,
        }
    }

    fn reset_kv_cache(&mut self) -> Result<()> {
        let cfg = Self::build_config_from_raw(&self.cfg_raw);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weight_files, DType::F32, &self.device)?
        };
        self.model = GPTBigCode::load(vb, cfg)?;
        Ok(())
    }

    fn resolve_weight_files(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<std::path::PathBuf>> {
        if let Ok(single) = repo.get("model.safetensors") {
            return Ok(vec![single]);
        }

        let index_path = repo
            .get("model.safetensors.index.json")
            .context("Neither model.safetensors nor sharded index found")?;
        let index_str = std::fs::read_to_string(index_path)?;
        let index: WeightsIndex = serde_json::from_str(&index_str)?;

        let mut seen = HashSet::new();
        let mut files = Vec::new();
        for shard in index.weight_map.values() {
            if seen.insert(shard.clone()) {
                files.push(repo.get(shard)?);
            }
        }
        files.sort();
        Ok(files)
    }

    fn logits_to_probs(&self, logits: Tensor) -> Result<Vec<f32>> {
        let next_logits = if logits.rank() == 2 {
            logits.i(0)?
        } else {
            let seq = logits.dim(1)?;
            logits.i((0, seq - 1))?
        };

        let probs_t = candle_nn::ops::softmax(&next_logits, 0)?;
        let mut probs: Vec<f32> = probs_t.to_vec1()?;

        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return Err(anyhow!("Model produced invalid probabilities"));
        }
        for p in &mut probs {
            *p = p.max(1e-9_f32) / sum;
        }
        Ok(probs)
    }

    // jak notebook: _reset_context(model, context_tokens)
    fn reset_context_probs(&mut self, context_tokens: &[u32]) -> Result<Vec<f32>> {
        self.reset_kv_cache()?;
        let input = Tensor::new(context_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        self.logits_to_probs(logits)
    }

    // jak notebook: _get_next_probs(..., past_kv)
    fn step_probs(&mut self, last_token: u32, kv_len: usize) -> Result<Vec<f32>> {
        let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, kv_len)?;
        self.logits_to_probs(logits)
    }

    pub fn compress(&mut self, text: &str) -> Result<Vec<u8>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!(e.to_string()))?;
        let tokens = encoding.get_ids();
        let num_tokens = tokens.len();
        let original_size = text.len() as u32;

        let mut encoder = DefaultRangeEncoder::new();

        // 1:1 z notebookiem
        let mut all_seen = vec![self.eos_token_id];
        let mut kv_len: usize = 0;

        for (i, &token) in tokens.iter().enumerate() {
            let probs = if kv_len >= CONTEXT_SIZE {
                let start = all_seen.len().saturating_sub(STRIDE);
                let context = &all_seen[start..];
                let probs = self.reset_context_probs(context)?;
                kv_len = context.len();
                probs
            } else {
                let last = *all_seen.last().unwrap_or(&self.eos_token_id);
                let probs = self.step_probs(last, kv_len)?;
                kv_len += 1;
                probs
            };

            let dist = DefaultLazyContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                &probs,
                None,
            )
            .map_err(|_| anyhow!("Failed to create categorical distribution"))?;
            encoder.encode_symbol(token as usize, dist.as_view())?;
            all_seen.push(token);

            if (i + 1) % 100 == 0 {
                println!("Encoded {}/{}", i + 1, num_tokens);
            }
        }

        let compressed_words = encoder.get_compressed();

        let mut output = Vec::new();
        output.write_u32::<LittleEndian>(original_size)?;
        output.write_u32::<LittleEndian>(num_tokens as u32)?;
        for &word in compressed_words.as_ref() {
            output.write_u32::<LittleEndian>(word)?;
        }

        Ok(output)
    }

    pub fn decompress(&mut self, compressed_data: &[u8]) -> Result<String> {
        let mut cursor = Cursor::new(compressed_data);
        let _original_size = cursor.read_u32::<LittleEndian>()?;
        let num_tokens = cursor.read_u32::<LittleEndian>()? as usize;

        let mut raw_data = Vec::new();
        while let Ok(word) = cursor.read_u32::<LittleEndian>() {
            raw_data.push(word);
        }

        let mut decoder = DefaultRangeDecoder::from_compressed(raw_data)?;

        // 1:1 z notebookiem
        let mut all_seen = vec![self.eos_token_id];
        let mut kv_len: usize = 0;
        let mut decoded_tokens = Vec::with_capacity(num_tokens);
        self.reset_kv_cache()?;

        for i in 0..num_tokens {
            let probs = if kv_len >= CONTEXT_SIZE {
                let start = all_seen.len().saturating_sub(STRIDE);
                let context = &all_seen[start..];
                let probs = self.reset_context_probs(context)?;
                kv_len = context.len();
                probs
            } else {
                let last = *all_seen.last().unwrap_or(&self.eos_token_id);
                let probs = self.step_probs(last, kv_len)?;
                kv_len += 1;
                probs
            };

            let dist = DefaultLazyContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                &probs,
                None,
            )
            .map_err(|_| anyhow!("Failed to create categorical distribution"))?;

            let token = decoder
                .decode_symbol(dist.as_view())
                .map_err(|_| anyhow!("Range decoding failed"))?;
            let token_u32 = token as u32;

            decoded_tokens.push(token_u32);
            all_seen.push(token_u32);

            if (i + 1) % 100 == 0 {
                println!("Decoded {}/{}", i + 1, num_tokens);
            }
        }

        let text = self
            .tokenizer
            .decode(&decoded_tokens, true)
            .map_err(|e| anyhow!(e.to_string()))?;
        Ok(text)
    }
}