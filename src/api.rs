// Copyright 2026 Mentat AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use axum::{
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tracing::info;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    pub model: Option<String>,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub response: String,
}

async fn health_check() -> impl IntoResponse {
    "OK"
}

async fn chat_handler(Json(payload): Json<ChatRequest>) -> impl IntoResponse {
    // In the future this will call the actual inference engine
    let response_text = format!("(Mock) Mentat received prompt: {}", payload.prompt);

    Json(ChatResponse {
        response: response_text,
    })
}

pub async fn start_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(chat_handler));

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!("Starting API server on {}", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
