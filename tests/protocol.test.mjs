import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const manifest = new URL('../protocol/manifest.json', import.meta.url);
const text = fs.readFileSync(manifest, 'utf8');
const data = JSON.parse(text);
execFileSync(process.execPath, ['protocol/validate_manifest.mjs'], {stdio:'pipe'});
assert.equal(data.protocol_version, 'phase1-2026-08-29');
const missing = JSON.parse(text); delete missing.evaluation.decoding.seed;
const temp = fileURLToPath(new URL('./bad-manifest.json', import.meta.url));
fs.writeFileSync(temp, JSON.stringify(missing));
try { assert.throws(() => execFileSync(process.execPath, ['protocol/validate_manifest.mjs', temp], {stdio:'pipe'}), /implicit decoding:seed/); }
finally { fs.rmSync(temp); }
console.log('3 protocol checks passed');
