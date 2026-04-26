//! ALCP digest (hash) primitives.

use aocl_crypto_sys as sys;
use aocl_error::{Error, Result};

/// Hash algorithms exposed by ALCP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Mode {
    Md5,
    Sha1,
    Sha2_224,
    Sha2_256,
    Sha2_384,
    Sha2_512,
    Sha2_512_224,
    Sha2_512_256,
    Sha3_224,
    Sha3_256,
    Sha3_384,
    Sha3_512,
}

impl Mode {
    fn raw(self) -> sys::alc_digest_mode_t {
        match self {
            Mode::Md5 => sys::_alc_digest_mode_ALC_MD5,
            Mode::Sha1 => sys::_alc_digest_mode_ALC_SHA1,
            Mode::Sha2_224 => sys::_alc_digest_mode_ALC_SHA2_224,
            Mode::Sha2_256 => sys::_alc_digest_mode_ALC_SHA2_256,
            Mode::Sha2_384 => sys::_alc_digest_mode_ALC_SHA2_384,
            Mode::Sha2_512 => sys::_alc_digest_mode_ALC_SHA2_512,
            Mode::Sha2_512_224 => sys::_alc_digest_mode_ALC_SHA2_512_224,
            Mode::Sha2_512_256 => sys::_alc_digest_mode_ALC_SHA2_512_256,
            Mode::Sha3_224 => sys::_alc_digest_mode_ALC_SHA3_224,
            Mode::Sha3_256 => sys::_alc_digest_mode_ALC_SHA3_256,
            Mode::Sha3_384 => sys::_alc_digest_mode_ALC_SHA3_384,
            Mode::Sha3_512 => sys::_alc_digest_mode_ALC_SHA3_512,
        }
    }

    /// Output size of this digest in bytes.
    pub const fn output_len(self) -> usize {
        match self {
            Mode::Md5 => 16,
            Mode::Sha1 => 20,
            Mode::Sha2_224 | Mode::Sha2_512_224 | Mode::Sha3_224 => 28,
            Mode::Sha2_256 | Mode::Sha2_512_256 | Mode::Sha3_256 => 32,
            Mode::Sha2_384 | Mode::Sha3_384 => 48,
            Mode::Sha2_512 | Mode::Sha3_512 => 64,
        }
    }
}

fn check_error(err: sys::alc_error_t) -> Result<()> {
    let is_err = unsafe { sys::alcp_is_error(err) };
    if is_err == 0 {
        return Ok(());
    }
    Err(Error::Status {
        component: "crypto",
        code: err as i64,
        message: format!("ALCP returned alc_error_t = {err:#x}"),
    })
}

/// Streaming hash context. Build with [`Digest::new`], feed chunks via
/// [`Digest::update`], then call [`Digest::finalize`] for the result.
pub struct Digest {
    handle: sys::alc_digest_handle_t,
    _context: Box<[u8]>,
    output_len: usize,
    finished: bool,
}

impl Digest {
    /// Create and initialize a new digest context for the given mode.
    pub fn new(mode: Mode) -> Result<Self> {
        let context_size = unsafe { sys::alcp_digest_context_size() } as usize;
        if context_size == 0 {
            return Err(Error::AllocationFailed("crypto"));
        }
        let mut context = vec![0u8; context_size].into_boxed_slice();
        let mut handle = sys::alc_digest_handle_t {
            context: context.as_mut_ptr() as *mut std::os::raw::c_void,
        };

        check_error(unsafe { sys::alcp_digest_request(mode.raw(), &mut handle) })?;
        check_error(unsafe { sys::alcp_digest_init(&mut handle) })?;

        Ok(Digest {
            handle,
            _context: context,
            output_len: mode.output_len(),
            finished: false,
        })
    }

    /// Feed a chunk of data into the digest.
    pub fn update(&mut self, data: &[u8]) -> Result<()> {
        if self.finished {
            return Err(Error::InvalidArgument(
                "update called after finalize".into(),
            ));
        }
        if data.is_empty() {
            return Ok(());
        }
        check_error(unsafe {
            sys::alcp_digest_update(&mut self.handle, data.as_ptr(), data.len() as u64)
        })
    }

    /// Finalize and return the digest. Consumes `self`.
    pub fn finalize(mut self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.output_len];
        let res = unsafe {
            sys::alcp_digest_finalize(
                &mut self.handle,
                out.as_mut_ptr(),
                out.len() as u64,
            )
        };
        check_error(res)?;
        self.finished = true;
        Ok(out)
    }
}

impl Drop for Digest {
    fn drop(&mut self) {
        unsafe { sys::alcp_digest_finish(&mut self.handle) };
    }
}

impl std::fmt::Debug for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Digest")
            .field("output_len", &self.output_len)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

/// One-shot convenience: hash `data` with the given mode.
pub fn hash(mode: Mode, data: &[u8]) -> Result<Vec<u8>> {
    let mut d = Digest::new(mode)?;
    d.update(data)?;
    d.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    #[test]
    fn sha256_empty() {
        let h = hash(Mode::Sha2_256, b"").unwrap();
        assert_eq!(
            hex(&h),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        let h = hash(Mode::Sha2_256, b"abc").unwrap();
        assert_eq!(
            hex(&h),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha512_abc() {
        let h = hash(Mode::Sha2_512, b"abc").unwrap();
        assert_eq!(
            hex(&h),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn streaming_matches_one_shot() {
        let payload = b"the quick brown fox jumps over the lazy dog";
        let one_shot = hash(Mode::Sha2_256, payload).unwrap();
        let mut d = Digest::new(Mode::Sha2_256).unwrap();
        d.update(&payload[..10]).unwrap();
        d.update(&payload[10..25]).unwrap();
        d.update(&payload[25..]).unwrap();
        let streamed = d.finalize().unwrap();
        assert_eq!(streamed, one_shot);
    }

    #[test]
    fn sha3_256_known_answer() {
        let h = hash(Mode::Sha3_256, b"abc").unwrap();
        assert_eq!(
            hex(&h),
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        );
    }

    #[test]
    fn output_len_matches_mode() {
        assert_eq!(hash(Mode::Sha2_256, b"x").unwrap().len(), 32);
        assert_eq!(hash(Mode::Sha2_512, b"x").unwrap().len(), 64);
        assert_eq!(hash(Mode::Sha3_384, b"x").unwrap().len(), 48);
    }
}
