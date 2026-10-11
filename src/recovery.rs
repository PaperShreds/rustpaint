//! Autosave and crash recovery.
//!
//! While there are unsaved changes, the app writes a recovery copy of the
//! project every `AUTOSAVE_INTERVAL` into a folder in the user's data
//! directory. A clean exit, or saving, removes it again; so a copy that is
//! still there on the next start means RustPaint didn't close properly.
//!
//! Several instances can run at once. Each one keeps a lock file locked while
//! it runs (the OS releases the lock if the process dies), and only recovery
//! copies whose lock is free are offered for restoring.

use crate::document::Document;
use crate::files;
use crate::project;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime};

pub const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(120);

const PROJECT_EXTENSION: &str = "rustpaint";
const INFO_EXTENSION: &str = "txt";
const LOCK_EXTENSION: &str = "lock";

/// What a recovery copy belongs to, for the restore prompt and afterwards.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Info {
    /// The project file the work belongs to, if it had been saved before.
    pub project_path: Option<PathBuf>,
    /// Name of the model, used to name exports.
    pub model_name: String,
}

impl Info {
    fn to_text(&self) -> String {
        let mut text = format!("model = {}\n", self.model_name);
        if let Some(path) = &self.project_path {
            text += &format!("project = {}\n", path.display());
        }
        text
    }

    fn from_text(text: &str) -> Self {
        let mut info = Self::default();
        for (key, value) in text.lines().filter_map(|line| line.split_once(" = ")) {
            match key {
                "model" => info.model_name = value.to_string(),
                "project" => info.project_path = Some(PathBuf::from(value)),
                _ => {}
            }
        }
        info
    }
}

/// This instance's recovery slot.
pub struct Session {
    project: PathBuf,
    info: PathBuf,
    lock_path: PathBuf,
    /// Held locked until the process ends; that marks the slot as in use.
    _lock: File,
    /// The autosave being written in the background, if any.
    writer: Option<JoinHandle<Result<(), String>>>,
}

/// A recovery copy left behind by an instance that didn't close properly.
pub struct Recovered {
    project: PathBuf,
    info_path: PathBuf,
    lock_path: PathBuf,
    pub info: Info,
    /// When it was written.
    pub saved_at: SystemTime,
}

impl Session {
    /// Claims a recovery slot in `directory`. Also returns the copies other
    /// instances left behind, newest first.
    pub fn start(directory: &Path) -> Result<(Self, Vec<Recovered>), String> {
        std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        let left_behind = find_left_behind(directory)?;

        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let stem = format!("session-{}-{nonce}", std::process::id());
        let lock_path = directory.join(format!("{stem}.{LOCK_EXTENSION}"));
        let lock = File::create(&lock_path).map_err(|e| e.to_string())?;
        lock.try_lock().map_err(|e| e.to_string())?;
        Ok((
            Self {
                project: directory.join(format!("{stem}.{PROJECT_EXTENSION}")),
                info: directory.join(format!("{stem}.{INFO_EXTENSION}")),
                lock_path,
                _lock: lock,
                writer: None,
            },
            left_behind,
        ))
    }

    /// Writes a recovery copy in the background. The document is encoded
    /// here; only the file writing happens on another thread. Returns false
    /// (and does nothing) if the previous autosave is still being written.
    pub fn autosave(&mut self, document: &Document, info: &Info) -> Result<bool, String> {
        if self
            .writer
            .as_ref()
            .is_some_and(|writer| !writer.is_finished())
        {
            return Ok(false);
        }
        self.finish_writing()?;
        let bytes = project::encode(document)?;
        let (project, info_path, info) = (self.project.clone(), self.info.clone(), info.to_text());
        self.writer = Some(std::thread::spawn(move || {
            files::write_atomically(&project, &bytes)?;
            files::write_atomically(&info_path, info.as_bytes())
        }));
        Ok(true)
    }

    /// Waits for a background autosave, and reports whether it failed.
    pub fn finish_writing(&mut self) -> Result<(), String> {
        match self.writer.take() {
            Some(writer) => writer
                .join()
                .unwrap_or_else(|_| Err("autosave thread failed".into())),
            None => Ok(()),
        }
    }

    /// Removes this session's recovery copy (the work is saved, or was discarded).
    pub fn clear(&mut self) {
        let _ = self.finish_writing();
        let _ = std::fs::remove_file(&self.project);
        let _ = std::fs::remove_file(&self.info);
    }

    /// Takes over a restored copy as this session's own, so it stays
    /// recoverable until the next autosave or save.
    pub fn adopt(&mut self, recovered: Recovered) -> Result<(), String> {
        self.finish_writing()?;
        std::fs::rename(&recovered.project, &self.project).map_err(|e| e.to_string())?;
        files::write_atomically(&self.info, recovered.info.to_text().as_bytes())?;
        recovered.discard();
        Ok(())
    }

    /// Clean exit: nothing is left to recover.
    pub fn close(mut self) {
        self.clear();
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

impl Recovered {
    pub fn load(&self) -> Result<Document, String> {
        project::load(&self.project)
    }

    /// Deletes the copy.
    pub fn discard(self) {
        let _ = std::fs::remove_file(&self.project);
        let _ = std::fs::remove_file(&self.info_path);
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

/// Recovery copies whose instance is gone (its lock file is no longer locked).
fn find_left_behind(directory: &Path) -> Result<Vec<Recovered>, String> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|e| e.to_string())? {
        let Ok(entry) = entry else {
            continue;
        };
        let lock_path = entry.path();
        if lock_path.extension().and_then(|ext| ext.to_str()) != Some(LOCK_EXTENSION) {
            continue;
        }
        let Ok(lock) = File::open(&lock_path) else {
            continue;
        };
        if lock.try_lock().is_err() {
            continue; // Another instance is running and owns it.
        }
        drop(lock);
        let project = lock_path.with_extension(PROJECT_EXTENSION);
        let info_path = lock_path.with_extension(INFO_EXTENSION);
        let saved_at = std::fs::metadata(&project).and_then(|m| m.modified());
        let Ok(saved_at) = saved_at else {
            // Nothing was autosaved (or it was cleaned up): just tidy away.
            let _ = std::fs::remove_file(&info_path);
            let _ = std::fs::remove_file(&lock_path);
            continue;
        };
        let info = std::fs::read_to_string(&info_path)
            .map(|text| Info::from_text(&text))
            .unwrap_or_default();
        found.push(Recovered {
            project,
            info_path,
            lock_path,
            info,
            saved_at,
        });
    }
    found.sort_by_key(|recovered| std::cmp::Reverse(recovered.saved_at));
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::primitives;

    fn temp_directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rustpaint-recovery-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn copies_are_offered_only_once_their_instance_is_gone() {
        let directory = temp_directory();
        let info = Info {
            project_path: Some(PathBuf::from("/projects/car = v2.rustpaint")),
            model_name: "car".into(),
        };
        let document = Document::new(primitives::cube());

        let (mut first, left_behind) = Session::start(&directory).unwrap();
        assert!(left_behind.is_empty());
        assert!(first.autosave(&document, &info).unwrap());
        first.finish_writing().unwrap();

        // The first instance is still running: nothing to recover.
        let (second, left_behind) = Session::start(&directory).unwrap();
        assert!(left_behind.is_empty());
        second.close();

        // The first instance "crashes" (its lock is released, files stay).
        drop(first);
        let (mut third, mut left_behind) = Session::start(&directory).unwrap();
        assert_eq!(left_behind.len(), 1);
        let recovered = left_behind.remove(0);
        assert_eq!(recovered.info, info);
        let restored = recovered.load().unwrap();
        assert_eq!(restored.mesh.vertices.len(), document.mesh.vertices.len());

        // Restoring makes it this instance's copy, still recoverable.
        third.adopt(recovered).unwrap();
        drop(third);
        let (fourth, left_behind) = Session::start(&directory).unwrap();
        assert_eq!(left_behind.len(), 1);
        assert_eq!(left_behind[0].info, info);
        for recovered in left_behind {
            recovered.discard();
        }
        fourth.close();

        // Everything was discarded or closed cleanly.
        let leftovers: Vec<_> = std::fs::read_dir(&directory).unwrap().collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn a_clean_exit_leaves_nothing_to_recover() {
        let directory = temp_directory();
        let (mut session, _) = Session::start(&directory).unwrap();
        let document = Document::new(primitives::cube());
        session.autosave(&document, &Info::default()).unwrap();
        session.close();
        let (next, left_behind) = Session::start(&directory).unwrap();
        assert!(left_behind.is_empty());
        next.close();
        std::fs::remove_dir(&directory).unwrap();
    }
}
