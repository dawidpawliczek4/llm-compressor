use std::sync::Arc;
use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use crate::AppState;


pub async fn compress_text(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Req>,
) -> Result<Json<Res>, (StatusCode, String)> {
    let mut compressor = state.compressor_service.lock().await;

    let compressed_text = compressor
        .compress(&payload.text)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(Res { compressed_text }))
}

pub async fn decompress_text(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DecompressReq>,
) -> Result<Json<DecompressRes>, (StatusCode, String)> {
    let mut compressor = state.compressor_service.lock().await;

    let text = compressor
        .decompress(&payload.compressed_text)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(DecompressRes { text }))
}


#[derive(Deserialize)]
pub struct Req {
    pub text: String,
}

#[derive(Serialize)]
pub struct Res {
    pub compressed_text: Vec<u8>,
}

#[derive(Deserialize)]
pub struct DecompressReq {
    pub compressed_text: Vec<u8>,
}

#[derive(Serialize)]
pub struct DecompressRes {
    pub text: String,
}