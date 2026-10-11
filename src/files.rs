//! File helpers shared by saving, exporting and autosave.

use std::io::Write as _;
use std::path::Path;

/// Writes `bytes` to a temporary file next to `path`, then renames it into
/// place. An interrupted write never leaves a half-written file behind, and
/// an existing file is only replaced once the new one is complete.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("path has no valid file name")?;
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));

    let mut created = false;
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        created = true;
        file.write_all(bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&temporary, path).map_err(|e| e.to_string())
    })();
    if result.is_err() && created {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}
