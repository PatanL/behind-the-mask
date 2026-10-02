// pump.fun's SDK (and Anchor under it) expect Node's Buffer; browsers don't have one. Load this before them.
import { Buffer } from 'buffer';
globalThis.Buffer ??= Buffer;
