//! Autosaves kept on purpose.
//!
//! Choosing "Don't save" when closing RustPaint or leaving a project doesn't
//! have to lose the work for good: a copy of it is kept here, and File >
//! Autosaves lists the latest few to open again. (Crash recovery is separate:
//! see `recovery`.)

use crate::document::Document;
use crate::files;
use crate::project;
use crate::recovery::Info;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How many autosaves are kept; the oldest go first.
pub const MAX_KEPT: usize = 10;

const PROJECT_EXTENSION: &str = "rustpaint";
const INFO_EXTENSION: &str = "txt";

/// One kept autosave.
pub struct Kept {
    project: PathBuf,
    info_path: PathBuf,
    pub info: Info,
    /// When it was kept.
    pub kept_at: SystemTime,
    /// Size of the project copy.
    pub bytes: u64,
}

impl Kept {
    pub fn path(&self) -> &Path {
        &self.project
    }

    pub fn load(&self) -> Result<Document, String> {
        project::load(&self.project)
    }

    pub fn delete(&self) {
        let _ = std::fs::remove_file(&self.project);
        let _ = std::fs::remove_file(&self.info_path);
    }
}

/// Keeps a copy of `document` in `directory`, and drops the oldest copies
/// beyond `MAX_KEPT`.
pub fn keep(directory: &Path, document: &Document, info: &Info) -> Result<(), String> {
    let bytes = project::encode(document)?;
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let mut millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    // Names are the time, so they sort by age; never reuse one.
    while directory
        .join(format!("{millis:016}.{PROJECT_EXTENSION}"))
        .exists()
    {
        millis += 1;
    }
    let project = directory.join(format!("{millis:016}.{PROJECT_EXTENSION}"));
    // The info first: a project file without one still lists, an info
    // without a project is cleaned up as a leftover.
    files::write_atomically(
        &project.with_extension(INFO_EXTENSION),
        info.to_text().as_bytes(),
    )?;
    files::write_atomically(&project, &bytes)?;
    for old in list(directory).iter().skip(MAX_KEPT) {
        old.delete();
    }
    Ok(())
}

/// The kept autosaves, newest first. Copies that can't be read are skipped.
pub fn list(directory: &Path) -> Vec<Kept> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut found: Vec<Kept> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some(PROJECT_EXTENSION))
        .filter_map(|project| {
            let millis: u64 = project.file_stem()?.to_str()?.parse().ok()?;
            let info_path = project.with_extension(INFO_EXTENSION);
            let info = std::fs::read_to_string(&info_path)
                .map(|text| Info::from_text(&text))
                .unwrap_or_default();
            Some(Kept {
                bytes: project.metadata().ok()?.len(),
                project,
                info_path,
                info,
                kept_at: UNIX_EPOCH + Duration::from_millis(millis),
            })
        })
        .collect();
    found.sort_by_key(|kept| std::cmp::Reverse(kept.kept_at));
    found
}

/// Deletes every kept autosave.
pub fn delete_all(directory: &Path) {
    for kept in list(directory) {
        kept.delete();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::primitives;

    fn directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("rustpaint-kept-{}-{nonce}", std::process::id()))
    }

    fn info(name: &str) -> Info {
        Info {
            project_path: Some(PathBuf::from(format!("/projects/{name}.rustpaint"))),
            model_name: name.to_string(),
        }
    }

    #[test]
    fn kept_copies_list_newest_first_and_open_again() {
        let directory = directory();
        let document = Document::new(primitives::cube());
        assert!(list(&directory).is_empty(), "no folder yet");
        keep(&directory, &document, &info("first")).unwrap();
        keep(&directory, &document, &info("second")).unwrap();
        let kept = list(&directory);
        let names: Vec<_> = kept.iter().map(|k| k.info.model_name.as_str()).collect();
        assert_eq!(names, ["second", "first"]);
        assert_eq!(kept[0].info, info("second"));
        assert!(kept[0].bytes > 0);
        assert!(kept[0].kept_at >= kept[1].kept_at);
        let opened = kept[0].load().unwrap();
        assert_eq!(opened.mesh.vertices.len(), document.mesh.vertices.len());

        kept[1].delete();
        assert_eq!(list(&directory).len(), 1);
        delete_all(&directory);
        assert!(list(&directory).is_empty());
        let leftovers: Vec<_> = std::fs::read_dir(&directory).unwrap().collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn only_the_latest_copies_are_kept() {
        let directory = directory();
        let document = Document::new(primitives::cube());
        for number in 0..MAX_KEPT + 3 {
            keep(&directory, &document, &info(&number.to_string())).unwrap();
        }
        let kept = list(&directory);
        assert_eq!(kept.len(), MAX_KEPT);
        assert_eq!(kept[0].info.model_name, (MAX_KEPT + 2).to_string());
        assert_eq!(kept[MAX_KEPT - 1].info.model_name, "3");
        // Pruned copies take their info file with them.
        let files = std::fs::read_dir(&directory).unwrap().count();
        assert_eq!(files, MAX_KEPT * 2);
        delete_all(&directory);
        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn stray_files_are_ignored() {
        let directory = directory();
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("notes.rustpaint"), b"x").unwrap();
        std::fs::write(directory.join("readme.txt"), b"x").unwrap();
        assert!(list(&directory).is_empty());
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
