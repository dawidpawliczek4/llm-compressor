
mod compressor;
use compressor::LLMCompressor;

use std::{env, sync::Arc};
use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

struct AppState {
    app_version: String,
}

#[derive(serde::Deserialize)]
struct Req {
    text: String,
}

#[derive(serde::Serialize)]
struct Res {
    compressed_text: Vec<u8>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let shared_state = Arc::new(AppState {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/compress", post(compress_text))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}


async fn health_check(State(state): State<Arc<AppState>>) -> String {
    format!("App version: {}", state.app_version)
}


async fn compress_text(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<Req>,
) -> std::result::Result<Json<Res>, (StatusCode, String)> {
    let mut compressor = LLMCompressor::new()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Compressor init failed: {e}")))?;
    
    let compressed_text = compressor
        .compress(&payload.text)
        .context("Failed to compress text")
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(Res { compressed_text }))
}
