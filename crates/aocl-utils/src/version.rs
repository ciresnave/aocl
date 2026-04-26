//! AOCL-Utils library version queries.
//!
//! AOCL-Utils declares `au_get_version_{major,minor,patch,str}` in its
//! public headers but the shipped Windows DLL does not export those
//! symbols (verified against `amd-utils/lib/libaoclutils.lib` in
//! AOCL 5.1). Until AMD ships them in the import lib, this module is
//! intentionally empty so it does not cause link errors. Users who need
//! the version can read `<AOCL_ROOT>/amd-utils/version.txt` (or the
//! release notes) directly.
