//! a-llama binary entry point.
//!
//! M1 wires the embedded knowledge store (Task 2), the inference engine
//! (Task 3), the GraphRAG augment loop (Task 4), the Eunomia cache (Task 5),
//! and the Ollama-compatible HTTP daemon (Task 6) into a single running
//! process.
//!
//! The default build uses the deterministic `MockEngine` (no model file, no
//! `mistralrs-engine` feature) so it boots instantly. The real mistral.rs GGUF
//! backend is feature-gated (Decision 7659) and selected at startup via env
//! vars (see [`build_state`]).
//!
//! ## Engine selection (env-var contract)
//!
//! Default (feature off, or `A_LLAMA_ENGINE` unset/not `mistralrs`):
//! deterministic [`MockEngine`], boots instantly, no model file.
//!
//! Real backend (built with `--features mistralrs-engine`, `A_LLAMA_ENGINE=mistralrs`):
//! - Chat GGUF: either `A_LLAMA_GGUF=/abs/path/to/model.gguf` (split into dir +
//!   file), or `A_LLAMA_MODEL_DIR` + `A_LLAMA_MODEL_FILE`.
//! - `A_LLAMA_CHAT_TEMPLATE` (optional): path to a chat-template JSON for GGUFs
//!   that don't embed one.
//! - `A_LLAMA_EMBED_MODEL` (optional): HF id of the 768-dim embedder; defaults
//!   to `google/embeddinggemma-300m` (Decision 7660).
//!
//! ## Runtime
//!
//! Uses the default multi-threaded `#[tokio::main]` runtime. The orchestrator's
//! GraphRAG path bridges the async engine to astraea-rag's synchronous provider
//! traits via `block_in_place`, which **requires** the multi-thread scheduler —
//! do NOT switch to `flavor = "current_thread"`.

use a_llama::api;

/// Address the daemon binds to. Override with `A_LLAMA_ADDR`. Defaults to
/// Ollama's port so existing clients work unchanged.
const DEFAULT_ADDR: &str = "127.0.0.1:11434";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr = std::env::var("A_LLAMA_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.to_string());

    // Select the inference engine and storage backend from the environment.
    let state = build_state().await?;
    let app = api::router(state.clone());

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!(
        "a-llama listening on http://{addr}  (model: {}, Ollama-compatible API)",
        state.model()
    );
    println!(
        "try:  curl http://{addr}/api/tags  |  \
         curl http://{addr}/api/generate -d '{{\"model\":\"{}\",\"prompt\":\"hi\",\"stream\":false}}'",
        state.model()
    );

    // Serve with graceful shutdown on SIGTERM/SIGINT.
    // In durable mode, the shutdown handler flushes the knowledge store and
    // cache to disk before the process exits so no WAL replay is needed on the
    // next startup.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;
    Ok(())
}

/// Wait for a SIGTERM or SIGINT, then flush durable state to disk (if enabled).
///
/// In non-durable builds this is just a ctrl-c handler with no persistence.
/// In durable builds (`--features durable`) it also calls
/// [`AppState::persist`] so the HNSW index and cache snapshot are written
/// before the process exits.
async fn shutdown_signal(_state: api::AppState) {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("a-llama: shutdown signal received");

    #[cfg(feature = "durable")]
    {
        eprintln!("a-llama: persisting durable state...");
        match _state.persist().await {
            Ok(()) => eprintln!("a-llama: durable state persisted successfully"),
            Err(e) => eprintln!("a-llama: warning: persist failed: {e}"),
        }
    }
}

/// Build the daemon's [`api::AppState`] over the engine selected by the
/// environment.
///
/// Defaults to the deterministic [`MockEngine`]. If the binary was built with
/// `--features mistralrs-engine` **and** `A_LLAMA_ENGINE=mistralrs`, loads the
/// real mistral.rs GGUF backend instead (see module docs for the env-var
/// contract). If the user asked for `mistralrs` but the feature is compiled
/// out, this warns loudly and falls back to mock rather than failing silently.
///
/// ## Durable mode (`--features durable`)
///
/// When the binary is built with `--features durable` **and**
/// `A_LLAMA_DATA_DIR` is set to an absolute path, the graph and cache are
/// backed by `DiskStorageEngine` and a WAL+snapshot cache at that directory.
/// Learned knowledge then survives process restarts.
///
/// ```text
/// A_LLAMA_DATA_DIR=/var/lib/a-llama ./a-llama   # durable mode
/// ```
///
/// The durable build still logs which mode is active.
async fn build_state() -> anyhow::Result<api::AppState> {
    let requested = std::env::var("A_LLAMA_ENGINE").unwrap_or_default();

    // Opt-in LLM fact/entity extraction on the cache-miss path (Task 9). It's an
    // extra LLM completion per miss, so it defaults off; enable with
    // A_LLAMA_EXTRACT_FACTS=1 (or `true`).
    let mut config = api::OrchestratorConfig::default();
    config.extract_facts = matches!(
        std::env::var("A_LLAMA_EXTRACT_FACTS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    );
    if config.extract_facts {
        println!("a-llama: fact extraction ENABLED (extra LLM call per cache miss)");
    }

    // 1. Select the inference engine — mock by default, real mistral.rs when the
    //    feature is compiled in and A_LLAMA_ENGINE=mistralrs.
    use a_llama::inference::InferenceEngine;
    let engine: std::sync::Arc<dyn InferenceEngine> = {
        #[cfg(feature = "mistralrs-engine")]
        {
            if requested == "mistralrs" {
                std::sync::Arc::new(build_mistralrs_engine().await?)
            } else {
                std::sync::Arc::new(a_llama::inference::MockEngine::new())
            }
        }
        #[cfg(not(feature = "mistralrs-engine"))]
        {
            if requested == "mistralrs" {
                eprintln!(
                    "warning: A_LLAMA_ENGINE=mistralrs was requested, but this binary was built \
                     WITHOUT the `mistralrs-engine` feature. Falling back to the mock engine. \
                     Rebuild with: cargo build --release --features mistralrs-engine"
                );
            }
            std::sync::Arc::new(a_llama::inference::MockEngine::new())
        }
    };
    let engine_label = engine.name().to_string();

    // 2. Select the storage backend — orthogonal to the engine choice. Durable
    //    (DiskStorageEngine + persisted cache) when built `--features durable`
    //    AND A_LLAMA_DATA_DIR is set; otherwise in-memory.
    #[cfg(feature = "durable")]
    if let Ok(data_dir_str) = std::env::var("A_LLAMA_DATA_DIR") {
        let data_dir = std::path::PathBuf::from(&data_dir_str);
        println!("a-llama engine: {engine_label} — durable mode");
        println!(
            "a-llama: durable storage at {} (graph WAL+HNSW under graph/, cache WAL+snapshot at root)",
            data_dir.display()
        );
        return api::AppState::open_durable(engine, &data_dir, config);
    }

    let state = api::AppState::with_config(engine, config)?;
    println!("a-llama engine: {engine_label} — in-memory mode");
    Ok(state)
}

/// Construct the real mistral.rs-backed [`api::AppState`] from the environment.
///
/// Resolves the chat GGUF from `A_LLAMA_GGUF` (a full file path) or the
/// `A_LLAMA_MODEL_DIR` + `A_LLAMA_MODEL_FILE` pair, plus the optional
/// `A_LLAMA_CHAT_TEMPLATE` and `A_LLAMA_EMBED_MODEL`. Any model-load failure is
/// propagated (so `main` exits non-zero) — it never silently falls back to mock
/// once the user opted into `mistralrs`.
#[cfg(feature = "mistralrs-engine")]
async fn build_mistralrs_engine() -> anyhow::Result<a_llama::inference::MistralRsEngine> {
    use a_llama::inference::MistralRsEngine;

    // Resolve the chat GGUF into (dir, file). Prefer a single absolute path.
    let (model_dir, model_file) = if let Ok(gguf) = std::env::var("A_LLAMA_GGUF") {
        let path = std::path::PathBuf::from(&gguf);
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        let file = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .ok_or_else(|| {
                anyhow::anyhow!("A_LLAMA_GGUF=`{gguf}` is not a file path (no filename component)")
            })?;
        (dir, file)
    } else {
        let dir = std::env::var("A_LLAMA_MODEL_DIR").map_err(|_| {
            anyhow::anyhow!(
                "mistralrs engine selected (A_LLAMA_ENGINE=mistralrs) but no chat model given. \
                 Set A_LLAMA_GGUF=/abs/path/to/model.gguf, or A_LLAMA_MODEL_DIR + A_LLAMA_MODEL_FILE."
            )
        })?;
        let file = std::env::var("A_LLAMA_MODEL_FILE").map_err(|_| {
            anyhow::anyhow!(
                "A_LLAMA_MODEL_DIR is set but A_LLAMA_MODEL_FILE is missing. \
                 Set A_LLAMA_MODEL_FILE to the GGUF filename inside A_LLAMA_MODEL_DIR."
            )
        })?;
        (dir, file)
    };

    let chat_template = std::env::var("A_LLAMA_CHAT_TEMPLATE").ok();
    let embed_model = std::env::var("A_LLAMA_EMBED_MODEL").ok();
    let embed_label = embed_model
        .clone()
        .unwrap_or_else(|| "google/embeddinggemma-300m".to_string());

    println!(
        "a-llama engine: mistralrs  (chat GGUF: {model_dir}/{model_file}, embedder: {embed_label}, 768-dim)"
    );
    if let Some(tpl) = &chat_template {
        println!("a-llama chat template: {tpl}");
    }
    println!("a-llama: loading model weights (this can take a while / may download)...");

    let engine = MistralRsEngine::load(
        model_dir,
        vec![model_file],
        chat_template,
        embed_model,
    )
    .await
    .map_err(|e| {
        anyhow::anyhow!(
            "failed to load mistralrs engine: {e}\n\
             check A_LLAMA_GGUF / A_LLAMA_MODEL_DIR+A_LLAMA_MODEL_FILE point at a valid GGUF, \
             and A_LLAMA_EMBED_MODEL (default google/embeddinggemma-300m) is a 768-dim embedder."
        )
    })?;

    println!("a-llama: mistralrs engine loaded successfully");
    Ok(engine)
}
