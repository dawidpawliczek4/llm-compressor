mod compressor;
use std::{env, sync::Arc};
use anyhow::Result;
use axum::{
    extract::State,
    routing::{get, post},
    Router,
};
use tokio::sync::Mutex;
use crate::compressor::service::LLMCompressor;
use crate::compressor::controller::{compress_text, decompress_text};


pub struct AppState {
    pub app_version: String,
    pub compressor_service: Mutex<LLMCompressor>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Trwa ładowanie modelu LLM do pamięci...");
    let compressor_service = LLMCompressor::new()?;
    println!("Model LLM załadowany!");

    let shared_state = Arc::new(AppState {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        compressor_service: Mutex::new(compressor_service),
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/compress", post(compress_text))
        .route("/decompress", post(decompress_text))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Serwer Axum nasłuchuje na -> http://0.0.0.0:3000");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check(State(state): State<Arc<AppState>>) -> String {
    format!("App version: {}", state.app_version)
}