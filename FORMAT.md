# WRPV version 10 format notes

These notes describe the WRPV v10 `.ngfx-gputrace` captures examined during
development.
All integers are little-endian. Sizes and flag meanings were checked against
the installed Nsight Graphics 2026.3 producer. Names for the two section flag
fields remain inferred, so the analyzer preserves their raw values.

## File header

| Offset | Type | Observed value | Meaning |
|---:|---|---:|---|
| `0x00` | `char[4]` | `WRPV` | File magic |
| `0x04` | `u32` | `10` | Container version |

One or more sections follow immediately and continue to end of file. There is
no observed file-level section count or table of contents.

## Section

Each section begins with 24 bytes (`<HHHHIIQ`):

| Offset | Type | Meaning |
|---:|---|---|
| `0x00` | `u16` | Section magic, `0x1234` |
| `0x02` | `u16` | Unknown flag A |
| `0x04` | `u16` | Unknown flag B |
| `0x06` | `u16` | Reserved, observed `0` |
| `0x08` | `u32` | Chunk count |
| `0x0c` | `u32` | Reserved, observed `0` |
| `0x10` | `u64` | Total uncompressed section size |

Observed section roles are:

| Flag A | Flag B | Payload |
|---:|---:|---|
| `1` | `0` | Serialized `NV.WarpViz.PbTraceData` protobuf |
| `0` | `1` | NVIDIA PerfWorks counter-data image (`LOPDATA\0`) |

## Chunk

The section header is followed by `chunk_count` inline chunk records. Each
record has a 24-byte header (`<HHIQQ`) followed immediately by its stored data:

| Offset | Type | Meaning |
|---:|---|---|
| `0x00` | `u16` | Chunk magic, `0x4321` |
| `0x02` | `u16` | Compression: `0` stored, `1` raw LZ4 block |
| `0x04` | `u32` | Reserved, observed `0` |
| `0x08` | `u64` | Stored byte count |
| `0x10` | `u64` | Uncompressed byte count |
| `0x18` | bytes | Stored payload |

The uncompressed sizes of all chunks must sum to the section's declared size.
Decompressed chunks are concatenated in record order to reconstruct a section.
In particular, the 35 chunks in each large section 1 form one PerfWorks image;
they are not independent `LOPDATA` images.

## Protobuf trace

Section 0 decodes as `NV.WarpViz.PbTraceData`. Nsight's WarpViz plugin contains
complete serialized `FileDescriptorProto` records, including `WarpViz.proto`,
shader profiler messages, and graphics event parameters. `TraceDocument` extracts
those descriptors at runtime and builds a dynamic protobuf descriptor pool.

Observed trace content includes:

- system/session information tables and process metadata;
- devices, command queues, command streams, API calls, and typed arguments;
- timestamp boundaries keyed by `NextCallIndex` and pipeline stage;
- swapchains, presented/application frames, and screenshots;
- NVTX, pipeline, semaphore, FECS, and hardware-event records;
- shader-profiler tables and compressed blobs;
- per-SM PC sampling storage and counter-data prefixes.

`NextCallIndex` partitions a command stream. The elapsed PTIMER value between
two adjacent boundaries applies to calls in `[previous.NextCallIndex,
current.NextCallIndex)`. The first boundary provides no start time for calls
before it.

## PerfWorks counter data

Section 1 begins with `LOPDATA\0` after concatenation and decompression. The
matching Nsight 2026.3 PerfWorks host library accepts it as a periodic-sampler
counter-data image. Across the samples it reports chip name `GB203`, populated
range counts from hundreds to tens of thousands, and complete ranges.

The following metadata is decoded through public PerfWorks entry points rather
than by treating internal offsets as stable:

- chip name and total range count;
- periodic sampler total, populated, and completed ranges;
- per-range timestamps, trigger counts, completeness, and descriptions.

The matching graphics metrics evaluator also maps named counter, ratio, and
throughput requests to each range. `PerfWorks::evaluate` uses that API directly,
while `PerfWorks::scan` evaluates one canonical form of every selected metric
base in a single pass.

## Real-time shader profiler (PC sampling)

These notes come from one Vulkan compute capture with samples (Nsight Graphics
2026.3.1, RTX 4080 / AD103, driver 610.47) and two Vulkan captures without
samples. Function addresses refer to the Nsight Graphics 2026.3.1 Windows
binaries and are given so the findings can be re-checked against later
releases. Each item states whether it is established (code or exact data
match) or inferred (consistent with the data but not traced in code).

### Per-SM sample storage

`gpus[].SamplingData.PcSamplingPerSM[]` holds one entry per virtual SM
(`vsmid`, `version`). Captures without samples have no entries. All observed
entries have `version` 1.

`pcSamplesStorage` is a sequence of 16-byte records:

| Offset | Type | Meaning | Status |
|---:|---|---|---|
| `0x00` | `u64` | Sampled PC as a GPU virtual address | established |
| `0x08` | `u32` | Observed `0` | — |
| `0x0c` | `u8` | Values 0–47, frequencies in groups of four; probably the warp slot | inferred |
| `0x0d` | `u8` | Sample type (hardware stall reason code), see below | established |
| `0x0e` | `u8` | Observed 0 or 1; `selected`/`not_selected` occur only with 0 | unknown |
| `0x0f` | `u8` | Observed `0` | — |

`timestampsStorage` is a sequence of 16-byte anchors `(u64 PTIMER, u64
cumulative sample index)`. Anchor `i` closes the samples
`[index[i-1], index[i])` of the same SM; the last index equals the record
count. Anchors are about 4 µs apart, and samples carry no individual
timestamps.

Nsight's sample total is smaller than the record count. In the AD103 capture
1 548 958 records exist, the UI reports 1 539 536, and 2 765 records fall
outside every code block. Counting only records whose anchor interval overlaps
the presented frame gives 1 539 480 (midpoint interpolation: 1 539 219). The
exact time window the UI applies is not yet established.

### Sample type codes

`WarpVizPlugin.dll` (`0x1800b4680`, switch at `0x1800b470c`) selects a code
table by `gpus[].gpuType.Arch`. Each entry is a
`smsp__pcsamp_warps_issue_stalled_<name>` metric name; the code is the table
index.

| Code | `Arch` 0x170, 0x190 | `Arch` 0x1A0, 0x1B0 |
|---:|---|---|
| 0 | misc | misc |
| 1 | drain | drain |
| 2 | imc_miss | barrier |
| 3 | barrier | membar |
| 4 | membar | sleeping |
| 5 | sleeping | branch_resolving |
| 6 | branch_resolving | wait |
| 7 | wait | long_scoreboard (RT core) |
| 8 | long_scoreboard (RT core) | long_scoreboard (L1TEX) |
| 9 | long_scoreboard (L1TEX) | short_scoreboard |
| 10 | short_scoreboard | no_instructions |
| 11 | no_instructions | math_pipe_throttle |
| 12 | math_pipe_throttle | tex_throttle |
| 13 | tex_throttle | mio_throttle (RT core) |
| 14 | mio_throttle (RT core) | lg_throttle |
| 15 | lg_throttle | mio_throttle (MIO) |
| 16 | mio_throttle (MIO) | dispatch_stall |
| 17 | dispatch_stall | selected |
| 18 | selected | not_selected |
| 19 | not_selected | misc |
| 20 | misc | — |

Any other architecture gets the single entry `misc`. The tables name both
long-scoreboard and both MIO-throttle codes identically. The pipe split is
inferred from the auto-exported `GPUTrace.PCSampler.tpc__warps_issue_stalled_*`
rows, which appear exactly in code order 1–20 and report zero for both RT-core
variants, matching codes 8 and 14 never occurring in the compute-only capture.
The AD103 capture reports `Arch` 400 (0x190) and uses the first table.

The exported `GPUTrace.PCSampler` stall percentages are not plain sample
shares: raw counts differ by up to 1.9 percentage points per reason. The
weighting Nsight applies is not yet established.

### Shader profiler blobs

`devices[].ShaderProfilerReport.blobTable.blobs[]` entries are referenced by
index (`PbBlobRef.idx`). `ShaderProfilerPlugin.dll` unpacks an entry in this
order (established, `0x1800645e0`):

1. `COMPRESSION_LZ4`: the data is one raw LZ4 block that expands to exactly
   `uncompressedSize` bytes (`0x1800502d0`).
2. `OBFUSCATION_SIMPLE`: the decompressed bytes are XORed with a ChaCha20
   keystream (`0x180316320`): original 20-round variant with constants
   `expand 32-byte k`, 64-bit block counter starting at 0, 64-bit nonce of
   zero bytes, and a fixed 32-byte key stored in `.rdata` of
   `ShaderProfilerPlugin.dll`, `WarpVizPlugin.dll` and
   `PylonGpuTraceFusionPlugin.dll`.

`ngfx-trace` does not ship the key. It tries every 4-byte aligned 32-byte
window of the schema binary as a key and keeps the one whose keystream turns
the first bytes of the capture's SPIR-V and cubin blobs into their known
headers.

All blobs of every examined capture unpack this way. Their contents:

| Reference | Content |
|---|---|
| `modules[].graphicsAPIModule.bytecode` | SPIR-V module as passed to the driver (`ILBytecodeType_SPIRV`) |
| `codeBlocks[].shaderInstance.instructions` | Raw SASS of the code block, `vaRange.size` bytes, 16 bytes per instruction |
| `codeBlocks[].shaderInstance.debugInfo.ref` | CUDA cubin (ELF64, machine 190, OS ABI 0x33) |

The cubin contains the same SASS in `.text.<function>`, plus `.debug_line`
(DWARF 2), `.debug_info` (producer `SPIRV Compiler`), `.nv.info` and constant
bank sections.

### Code block to module and source line

Each cubin's `.debug_line` has one include directory of
`"`-separated `key>value` pairs, for example
`./"stage>cs"fmt>spir"ShaderModuleHash>36cf63efe2e621e`, and one file such as
`comp.10000.spv`. Nsight parses the pairs and reads `ShaderModuleHash` as hex
(`0x1801b74b0`).

`ShaderModuleHash` is Bob Jenkins' 64-bit `lookup8` hash (initial level 0)
over the unpacked SPIR-V bytes (`0x180238470`). For all twelve distinct
modules of the examined captures the hash equals the value in the cubin, and
for the capture with samples it also equals the shader hash shown in the UI.
This is the only link from a code block to its module; the protobuf has no
module index.

The line program maps SASS offsets (relative to `vaRange.address`) to a
"line" in the virtual `.spv` file. That line is the zero-based index of a
SPIR-V instruction plus 6, i.e. the line in a `spirv-dis` listing with a
five-line header (inferred: at this offset every mapped line in four checked
modules lands on a non-debug instruction; at any other offset 30–115 lines
land on `OpLine`, `OpLabel` or similar). The source file and line then come
from the SPIR-V `OpLine` in effect for that instruction.

Nsight keeps that line across `OpNoLine`, unlike the SPIR-V rule: in
`detail_still.comp` an `OpSelect` after `OpNoLine` carries 0.19 % of all
samples, and the UI counts them for the preceding line 119. Whether the line
also survives `OpLabel` is not yet established; `ngfx-trace` ends it there.
Restricted to the presented frame, this attribution reproduces all nine UI
hotspot lines and the UI's per-instruction split of `detail_still.comp:106`
to 0.01 percentage points.

For `NonSemantic.Shader.DebugInfo.100` modules (`glslang -gVS`) `ngfx-trace`
uses `DebugLine`, `DebugSource`, and `DebugCompilationUnit` the same way. The
Nsight UI does not use this debug information, so this path has no UI
reference.

The UI's own SPIR-V line numbers differ (for example UI line 1402 for
instruction index 564 of `detail_still.comp`); its listing apparently also
counts embedded source text.

## Confidence and unknowns

Confirmed on all six samples:

- the v10 file/section/chunk headers and LZ4 mode;
- section concatenation and exact uncompressed sizes;
- the section 0 protobuf type and descriptor source;
- section 1 as one valid PerfWorks periodic counter image.
- arbitrary named metric evaluation and complete catalog scanning.

Confirmed on the three Vulkan captures from Nsight Graphics 2026.3.1:

- the shader profiler blob encoding (LZ4 block, then ChaCha20) and the
  SPIR-V/SASS/cubin blob contents;
- `ShaderModuleHash` as `lookup8` over the SPIR-V module;
- the 16-byte PC sample and timestamp anchor records (one capture with
  samples);
- the per-architecture sample type tables in the 2026.3.1 WarpViz plugin.

Not yet established:

- compatibility with WRPV versions other than 10;
- official names or broader meanings of the section flag fields;
- other compression identifiers or section roles;
- the full set of proprietary `GPUTrace.*` formulas behind every UI row,
  including the weighting of the exported PC sampler stall percentages;
- the meaning of PC sample bytes `0x0c` and `0x0e`;
- the exact time window behind the UI's sample total;
- whether the UI carries source lines across `OpLabel`;
- the UI's SPIR-V line numbering;
- the SASS instruction classes behind the UI's instruction mix;
- PC sampling on architectures other than Ada and on D3D12 or OpenGL.

The parser rejects unvalidated WRPV versions instead of assuming the v10
layout. It also retains unknown protobuf fields when `trace.pb` is extracted,
allowing future schema work without lossy conversion.
