use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::repo::RepoSnapshot;
use crate::repo::scan_repo;

#[derive(Debug, Clone)]
pub struct SessionStore {
    inner: Arc<RwLock<BTreeMap<String, Session>>>,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }
}

#[derive(Debug, Clone)]
struct Session {
    id: String,
    label: Option<String>,
    repo_root: PathBuf,
    active_stage: Option<String>,
    pinned_terms: Vec<String>,
    invariants: Vec<String>,
    notes: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    snapshot: RepoSnapshot,
}

#[derive(Debug, Clone)]
pub struct SessionSeed {
    pub label: Option<String>,
    pub repo_root: PathBuf,
    pub active_stage: Option<String>,
    pub pinned_terms: Vec<String>,
    pub invariants: Vec<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionUpdate {
    pub session_id: String,
    pub label: Option<String>,
    pub active_stage: Option<String>,
    pub pinned_terms: Option<Vec<String>>,
    pub invariants: Option<Vec<String>>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionView {
    pub session_id: String,
    pub label: Option<String>,
    pub repo_root: String,
    pub repo_name: String,
    pub snapshot_id: String,
    pub active_stage: Option<String>,
    pub pinned_terms: Vec<String>,
    pub invariants: Vec<String>,
    pub notes: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub indexed_at: String,
}

impl SessionStore {
    pub fn create(&self, seed: SessionSeed) -> Result<SessionView> {
        let snapshot = scan_repo(&seed.repo_root)?;
        let now = Utc::now();
        let session = Session {
            id: format!("sess-{}", Uuid::new_v4().simple()),
            label: seed.label,
            repo_root: seed.repo_root,
            active_stage: seed.active_stage,
            pinned_terms: seed.pinned_terms,
            invariants: seed.invariants,
            notes: seed.notes,
            created_at: now,
            updated_at: now,
            snapshot,
        };
        let view = session.to_view();
        self.inner
            .write()
            .map_err(|_| anyhow!("session store poisoned"))?
            .insert(session.id.clone(), session);
        Ok(view)
    }

    pub fn list(&self) -> Result<Vec<SessionView>> {
        let store = self
            .inner
            .read()
            .map_err(|_| anyhow!("session store poisoned"))?;
        Ok(store.values().map(Session::to_view).collect())
    }

    pub fn get(&self, session_id: &str) -> Result<SessionView> {
        let store = self
            .inner
            .read()
            .map_err(|_| anyhow!("session store poisoned"))?;
        let session = store
            .get(session_id)
            .with_context(|| format!("unknown session: {session_id}"))?;
        Ok(session.to_view())
    }

    pub fn snapshot(&self, session_id: &str) -> Result<RepoSnapshot> {
        let store = self
            .inner
            .read()
            .map_err(|_| anyhow!("session store poisoned"))?;
        let session = store
            .get(session_id)
            .with_context(|| format!("unknown session: {session_id}"))?;
        Ok(session.snapshot.clone())
    }

    pub fn effective_terms(&self, session_id: &str) -> Result<Vec<String>> {
        let store = self
            .inner
            .read()
            .map_err(|_| anyhow!("session store poisoned"))?;
        let session = store
            .get(session_id)
            .with_context(|| format!("unknown session: {session_id}"))?;
        Ok(session.pinned_terms.clone())
    }

    pub fn update(&self, update: SessionUpdate) -> Result<SessionView> {
        let mut store = self
            .inner
            .write()
            .map_err(|_| anyhow!("session store poisoned"))?;
        let session = store
            .get_mut(&update.session_id)
            .with_context(|| format!("unknown session: {}", update.session_id))?;

        session.label = update.label;
        session.active_stage = update.active_stage;
        if let Some(pinned_terms) = update.pinned_terms {
            session.pinned_terms = pinned_terms;
        }
        if let Some(invariants) = update.invariants {
            session.invariants = invariants;
        }
        session.notes = update.notes;
        session.updated_at = Utc::now();
        Ok(session.to_view())
    }

    pub fn reset(&self, session_id: &str) -> Result<SessionView> {
        let mut store = self
            .inner
            .write()
            .map_err(|_| anyhow!("session store poisoned"))?;
        let session = store
            .get_mut(session_id)
            .with_context(|| format!("unknown session: {session_id}"))?;
        session.snapshot = scan_repo(&session.repo_root)?;
        session.updated_at = Utc::now();
        Ok(session.to_view())
    }

    pub fn drop_session(&self, session_id: &str) -> Result<SessionView> {
        let mut store = self
            .inner
            .write()
            .map_err(|_| anyhow!("session store poisoned"))?;
        let session = store
            .remove(session_id)
            .with_context(|| format!("unknown session: {session_id}"))?;
        Ok(session.to_view())
    }
}

impl Session {
    fn to_view(&self) -> SessionView {
        SessionView {
            session_id: self.id.clone(),
            label: self.label.clone(),
            repo_root: self.repo_root.display().to_string(),
            repo_name: self.snapshot.repo_name.clone(),
            snapshot_id: self.snapshot.snapshot_id.clone(),
            active_stage: self.active_stage.clone(),
            pinned_terms: self.pinned_terms.clone(),
            invariants: self.invariants.clone(),
            notes: self.notes.clone(),
            created_at: self.created_at.to_rfc3339(),
            updated_at: self.updated_at.to_rfc3339(),
            indexed_at: self.snapshot.indexed_at.to_rfc3339(),
        }
    }
}
