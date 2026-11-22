//! Opening a dataset folder: the `*.csv` files of events/, clicks/ and links/ mapped read-only, the
//! small files read whole.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::records::Locator;
use crate::tables::{self, Campaigns, Ctx, Fallback, Products};

#[allow(non_camel_case_types)]
type c_void = core::ffi::c_void;

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_POPULATE: i32 = 0x8000;
pub const MADV_SEQUENTIAL: i32 = 2;
pub const MADV_WILLNEED: i32 = 3;
pub const MADV_HUGEPAGE: i32 = 14;
pub const MADV_POPULATE_READ: i32 = 22;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapMode {
    /// Plain mmap; pages fault in on first touch.
    Lazy,
    /// mmap with MAP_POPULATE.
    Populate,
}

/// A read-only private mapping of a whole file. Empty files map to an empty slice.
pub struct Mapped {
    ptr: *mut u8,
    len: usize,
    pub path: PathBuf,
}

unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

impl Mapped {
    pub fn open(path: &Path, mode: MapMode) -> io::Result<Mapped> {
        use std::os::fd::AsRawFd;
        let file = fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mapped { ptr: std::ptr::null_mut(), len: 0, path: path.to_path_buf() });
        }
        let flags = MAP_PRIVATE | if mode == MapMode::Populate { MAP_POPULATE } else { 0 };
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, flags, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Mapped { ptr: ptr as *mut u8, len, path: path.to_path_buf() })
    }

    #[inline(always)]
    pub fn bytes(&self) -> &[u8] {
        if self.len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }
    }

    /// madvise over a byte range of the mapping (rounded out to pages). Returns the raw result.
    pub fn advise(&self, from: usize, to: usize, advice: i32) -> i32 {
        if self.len == 0 || from >= to {
            return 0;
        }
        let page = 4096usize;
        let start = from & !(page - 1);
        let end = to.min(self.len);
        unsafe { madvise(self.ptr.add(start) as *mut c_void, end - start, advice) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        if self.len != 0 {
            unsafe { munmap(self.ptr as *mut c_void, self.len) };
        }
    }
}

/// The `*.csv` files directly in `folder`, sorted by name (this order gives file numbers).
/// Names starting with `.` are skipped, as a shell glob would.
pub fn csv_files(folder: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !folder.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(folder)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || !name.ends_with(".csv") {
            continue;
        }
        if fs::metadata(entry.path())?.is_file() {
            out.push(entry.path());
        }
    }
    out.sort();
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stream {
    Events = 0,
    Clicks = 1,
    Links = 2,
}

pub struct Dataset {
    pub dir: PathBuf,
    /// Indexed by Stream, then by file number.
    pub files: [Vec<Mapped>; 3],
    pub products: Vec<u8>,
    pub campaigns: Vec<u8>,
    pub manifest: Vec<u8>,
}

impl Dataset {
    pub fn open(dir: &Path, mode: MapMode) -> io::Result<Dataset> {
        let map_all = |sub: &str| -> io::Result<Vec<Mapped>> {
            csv_files(&dir.join(sub))?.iter().map(|p| Mapped::open(p, mode)).collect()
        };
        Ok(Dataset {
            dir: dir.to_path_buf(),
            files: [map_all("events")?, map_all("clicks")?, map_all("links")?],
            products: fs::read(dir.join("products.csv")).unwrap_or_default(),
            campaigns: fs::read(dir.join("campaigns.csv")).unwrap_or_default(),
            manifest: fs::read(dir.join("manifest.json")).unwrap_or_default(),
        })
    }

    pub fn stream(&self, s: Stream) -> &[Mapped] {
        &self.files[s as usize]
    }

    pub fn bytes(&self, s: Stream) -> u64 {
        self.files[s as usize].iter().map(|m| m.len() as u64).sum()
    }

    /// The bytes of the line a locator points at.
    pub fn line(&self, s: Stream, at: Locator) -> &[u8] {
        crate::reference::line_at(self.files[s as usize][at.file() as usize].bytes(), at.offset())
    }

    /// Checks the dataset-level limits of the locators.
    pub fn check_limits(&self) -> Result<(), Fallback> {
        for (s, name) in [(Stream::Events, "events"), (Stream::Clicks, "clicks"), (Stream::Links, "links")] {
            let files = self.stream(s);
            if files.len() > Locator::MAX_FILES {
                return Err(Fallback::TooManyFiles(name, files.len()));
            }
            if let Some(f) = files.iter().find(|f| f.len() as u64 >= Locator::MAX_OFFSET) {
                return Err(Fallback::FileTooLarge(f.path.display().to_string()));
            }
        }
        Ok(())
    }

    pub fn tables(&self) -> Result<((i64, i64), Products, Campaigns), Fallback> {
        Ok((tables::manifest(&self.manifest)?, tables::products(&self.products)?, tables::campaigns(&self.campaigns)?))
    }

    /// The parse context (window and lookup tables).
    pub fn ctx(&self) -> Result<Ctx, Fallback> {
        self.check_limits()?;
        let (window, products, campaigns) = self.tables()?;
        Ctx::new(window, products, campaigns)
    }
}
