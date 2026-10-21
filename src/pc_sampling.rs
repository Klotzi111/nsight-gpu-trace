//! Real-time shader profiler PC samples: per-SM records, sample type tables,
//! anchor-interpolated timestamps, and attribution to shaders, SASS, SPIR-V,
//! and source lines through [`ShaderProfile`].

use std::collections::{BTreeMap, HashMap};

use regex::Regex;
use serde::Serialize;

use crate::{
    Error, Result, Scope, ScopeKind, TraceDocument,
    analysis::{message_items, optional_bytes, optional_message, optional_string, optional_u64},
    shader_profiler::{
        CodeBlock, InstructionClass, SPIRV_FIRST_INSTRUCTION_LINE, ShaderProfile, file_basename,
        format_hash,
    },
};

const RECORD_SIZE: usize = 16;
const ANCHOR_SIZE: usize = 16;
const SUPPORTED_STORAGE_VERSION: u64 = 1;
/// Samples carry no individual timestamps; each one gets the midpoint of its
/// slot between the two per-SM anchors that enclose it.
pub const TIME_ATTRIBUTION: &str = "midpoint_between_per_sm_anchors";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleKind {
    Stalled,
    Issued,
    Eligible,
    Unknown,
}

/// One hardware sample type (stall reason) code.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SampleType {
    pub code: u8,
    /// Suffix of the exported `tpc__warps_issue_stalled_*` metric.
    pub name: &'static str,
    pub label: &'static str,
    pub kind: SampleKind,
}

const fn stalled(code: u8, name: &'static str, label: &'static str) -> SampleType {
    SampleType {
        code,
        name,
        label,
        kind: SampleKind::Stalled,
    }
}

const SELECTED: &str = "selected";
const NOT_SELECTED: &str = "not_selected";

/// WarpViz table for `gpuType.Arch` 0x170 (Ampere) and 0x190 (Ada).
const AMPERE_ADA_TYPES: [SampleType; 21] = [
    stalled(0, "misc", "Miscellaneous"),
    stalled(1, "drain", "Drain"),
    stalled(2, "imc_miss", "IMC Miss"),
    stalled(3, "barrier", "Barrier"),
    stalled(4, "membar", "Memory Barrier"),
    stalled(5, "sleeping", "Sleeping"),
    stalled(6, "branch_resolving", "Branch Resolving"),
    stalled(7, "wait", "Wait"),
    stalled(
        8,
        "long_scoreboard_pipe_rtcore",
        "Long Scoreboard (RT Core)",
    ),
    stalled(9, "long_scoreboard_pipe_l1tex", "Long Scoreboard (L1TEX)"),
    stalled(10, "short_scoreboard", "Short Scoreboard"),
    stalled(11, "no_instruction", "No Instruction"),
    stalled(12, "math_pipe_throttle", "Math Pipe Throttle"),
    stalled(13, "tex_throttle", "Texture Throttle"),
    stalled(14, "mio_throttle_pipe_rtcore", "MIO Throttle (RT Core)"),
    stalled(15, "lg_throttle", "LG Throttle"),
    stalled(16, "mio_throttle_pipe_mio", "MIO Throttle"),
    stalled(17, "dispatch_stall", "Dispatch Stall"),
    SampleType {
        code: 18,
        name: SELECTED,
        label: "Selected",
        kind: SampleKind::Issued,
    },
    SampleType {
        code: 19,
        name: NOT_SELECTED,
        label: "Not Selected",
        kind: SampleKind::Eligible,
    },
    stalled(20, "misc", "Miscellaneous"),
];

/// WarpViz table for `gpuType.Arch` 0x1A0 and 0x1B0 (Blackwell).
const BLACKWELL_TYPES: [SampleType; 20] = [
    stalled(0, "misc", "Miscellaneous"),
    stalled(1, "drain", "Drain"),
    stalled(2, "barrier", "Barrier"),
    stalled(3, "membar", "Memory Barrier"),
    stalled(4, "sleeping", "Sleeping"),
    stalled(5, "branch_resolving", "Branch Resolving"),
    stalled(6, "wait", "Wait"),
    stalled(
        7,
        "long_scoreboard_pipe_rtcore",
        "Long Scoreboard (RT Core)",
    ),
    stalled(8, "long_scoreboard_pipe_l1tex", "Long Scoreboard (L1TEX)"),
    stalled(9, "short_scoreboard", "Short Scoreboard"),
    stalled(10, "no_instruction", "No Instruction"),
    stalled(11, "math_pipe_throttle", "Math Pipe Throttle"),
    stalled(12, "tex_throttle", "Texture Throttle"),
    stalled(13, "mio_throttle_pipe_rtcore", "MIO Throttle (RT Core)"),
    stalled(14, "lg_throttle", "LG Throttle"),
    stalled(15, "mio_throttle_pipe_mio", "MIO Throttle"),
    stalled(16, "dispatch_stall", "Dispatch Stall"),
    SampleType {
        code: 17,
        name: SELECTED,
        label: "Selected",
        kind: SampleKind::Issued,
    },
    SampleType {
        code: 18,
        name: NOT_SELECTED,
        label: "Not Selected",
        kind: SampleKind::Eligible,
    },
    stalled(19, "misc", "Miscellaneous"),
];

/// Select the sample type table WarpViz uses for an architecture ID.
pub fn sample_type_table(arch: u64) -> Option<(&'static str, &'static [SampleType])> {
    match arch {
        0x170 => Some(("ampere", &AMPERE_ADA_TYPES)),
        0x190 => Some(("ada", &AMPERE_ADA_TYPES)),
        0x1a0 | 0x1b0 => Some(("blackwell", &BLACKWELL_TYPES)),
        _ => None,
    }
}

/// One decoded 16-byte PC sample record.
#[derive(Debug, Clone, Copy)]
pub struct PcSample {
    pub gpu_index: u16,
    pub vsmid: u32,
    pub pc: u64,
    pub time_ns: Option<u64>,
    /// Byte `0x0c`, 0-47 on Ada; probably the warp slot (not established).
    pub warp_slot: u8,
    pub sample_type: u8,
    /// Byte `0x0e`, observed 0 or 1 (meaning not established).
    pub flag: u8,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmStream {
    pub gpu_index: usize,
    pub vsmid: u32,
    pub version: Option<u64>,
    pub records: usize,
    pub anchors: usize,
    pub dropped_storage_bytes: usize,
    pub first_ptimer: Option<u64>,
    pub last_ptimer: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuSamplingInfo {
    pub gpu_index: usize,
    pub arch: Option<u64>,
    pub implementation: Option<u64>,
    pub chip_name: Option<String>,
    pub sampling_interval: Option<u64>,
    pub sample_type_table: Option<&'static str>,
}

/// All PC samples of a trace plus the shader tables needed to attribute them.
#[derive(Debug, Clone)]
pub struct PcSampling {
    pub gpus: Vec<GpuSamplingInfo>,
    pub streams: Vec<SmStream>,
    samples: Vec<PcSample>,
    sample_types: &'static [SampleType],
    profile: ShaderProfile,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleTypeCount {
    pub code: u8,
    pub name: String,
    pub label: String,
    pub kind: SampleKind,
    pub samples: usize,
    pub share_pct: f64,
}

/// Samples of one instruction mix class; `None` fields mark unclassified samples.
#[derive(Debug, Clone, Serialize)]
pub struct InstructionClassCount {
    pub label: String,
    pub pipe: Option<&'static str>,
    pub family: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<&'static str>,
    pub samples: usize,
    pub share_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AttributionCounts {
    pub in_code_blocks: usize,
    pub outside_code_blocks: usize,
    pub with_spirv_instruction: usize,
    pub with_source_line: usize,
    pub with_instruction_class: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShaderSamples {
    pub shader: String,
    pub hash: Option<String>,
    pub code_blocks: Vec<usize>,
    pub samples: usize,
    pub share_pct: f64,
    pub sample_types: Vec<SampleTypeCount>,
    pub instruction_mix: Vec<InstructionClassCount>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PcSamplingSummary {
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<&'static str>,
    pub gpus: Vec<GpuSamplingInfo>,
    pub records: usize,
    pub sm_streams: usize,
    pub anchors: usize,
    pub scope: Scope,
    pub selected_samples: usize,
    pub untimed_samples_excluded: usize,
    pub attribution: AttributionCounts,
    pub sample_types: Vec<SampleTypeCount>,
    pub instruction_mix: Vec<InstructionClassCount>,
    pub shaders: Vec<ShaderSamples>,
    pub time_attribution: &'static str,
    pub exact_per_sample_timestamps: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HotspotGrouping {
    Line,
    Instruction,
    Sass,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hotspot {
    pub shader: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shader_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spirv_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spirv_listing_line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spirv_instruction: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_block: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sass_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sass_encoding: Option<String>,
    pub samples: usize,
    pub share_pct: f64,
    pub sample_types: Vec<SampleTypeCount>,
    pub instruction_mix: Vec<InstructionClassCount>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HotspotReport {
    pub group_by: HotspotGrouping,
    pub scope: Scope,
    pub selected_samples: usize,
    pub matching_samples: usize,
    pub hotspots: Vec<Hotspot>,
    pub share_basis: &'static str,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleRecord {
    pub index: usize,
    pub gpu_index: u16,
    pub vsmid: u32,
    pub pc: u64,
    pub time_ns: Option<u64>,
    pub sample_type: u8,
    pub sample_type_name: String,
    pub warp_slot: u8,
    pub flag: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shader: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shader_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_block: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sass_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spirv_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

/// Per-group sample counts by sample type and instruction mix class.
struct GroupCounts {
    types: Box<[usize; 256]>,
    classes: HashMap<Option<InstructionClass>, usize>,
}

impl GroupCounts {
    fn new() -> Self {
        Self {
            types: Box::new([0; 256]),
            classes: HashMap::new(),
        }
    }

    fn add(&mut self, sample_type: u8, class: Option<InstructionClass>) {
        self.types[usize::from(sample_type)] += 1;
        *self.classes.entry(class).or_default() += 1;
    }

    fn total(&self) -> usize {
        self.types.iter().sum()
    }
}

/// Samples chosen by a scope, as indices into the decoded sample list.
struct Selection {
    scope: Scope,
    indices: Vec<usize>,
    untimed: usize,
    warnings: Vec<String>,
}

impl PcSampling {
    pub fn from_document(document: &TraceDocument) -> Result<Self> {
        let mut gpus = Vec::new();
        let mut streams = Vec::new();
        let mut samples = Vec::new();
        let mut warnings = Vec::new();
        let mut sample_types: Option<&'static [SampleType]> = None;
        for (gpu_index, gpu) in message_items(document.message(), "gpus")?.enumerate() {
            let gpu_type = optional_message(gpu, "gpuType")?;
            let arch = gpu_type.and_then(|value| optional_u64(value, "Arch"));
            let table = arch.and_then(sample_type_table);
            let Some(sampling) = optional_message(gpu, "SamplingData")? else {
                continue;
            };
            gpus.push(GpuSamplingInfo {
                gpu_index,
                arch,
                implementation: gpu_type.and_then(|value| optional_u64(value, "Impl")),
                chip_name: optional_string(sampling, "ChipName"),
                sampling_interval: optional_u64(sampling, "pcSamplingInterval"),
                sample_type_table: table.map(|table| table.0),
            });
            let per_sm: Vec<_> = message_items(sampling, "PcSamplingPerSM")?.collect();
            if per_sm.is_empty() {
                continue;
            }
            match table {
                Some((_, table)) if sample_types.is_none_or(|current| current.len() == table.len()) => {
                    sample_types = Some(table)
                }
                Some(_) => warnings.push(
                    "GPUs with different sample type tables; codes use the first GPU's table."
                        .into(),
                ),
                None => warnings.push(format!(
                    "GPU {gpu_index} architecture {} has no known sample type table; codes are reported without names.",
                    arch.map_or_else(|| "unknown".into(), |arch| format!("{arch:#x}"))
                )),
            }
            for stream in per_sm {
                let vsmid = optional_u64(stream, "vsmid").unwrap_or(0) as u32;
                let version = optional_u64(stream, "version");
                let records = optional_bytes(stream, "pcSamplesStorage").unwrap_or_default();
                let timestamps = optional_bytes(stream, "timestampsStorage").unwrap_or_default();
                let dropped =
                    optional_bytes(stream, "droppedPcSamplesStorage").map_or(0, <[u8]>::len);
                let mut info = SmStream {
                    gpu_index,
                    vsmid,
                    version,
                    records: 0,
                    anchors: 0,
                    dropped_storage_bytes: dropped,
                    first_ptimer: None,
                    last_ptimer: None,
                };
                if version != Some(SUPPORTED_STORAGE_VERSION) {
                    warnings.push(format!(
                        "SM {vsmid} of GPU {gpu_index} uses unsupported PC sample storage version {version:?}; it was skipped."
                    ));
                    streams.push(info);
                    continue;
                }
                if !records.len().is_multiple_of(RECORD_SIZE) {
                    warnings.push(format!(
                        "SM {vsmid} of GPU {gpu_index} has {} sample bytes, not a multiple of {RECORD_SIZE}; it was skipped.",
                        records.len()
                    ));
                    streams.push(info);
                    continue;
                }
                let count = records.len() / RECORD_SIZE;
                let anchors = match decode_anchors(timestamps, count) {
                    Ok(anchors) => anchors,
                    Err(error) => {
                        warnings.push(format!(
                            "SM {vsmid} of GPU {gpu_index}: {error}; its samples have no time."
                        ));
                        Vec::new()
                    }
                };
                let times = sample_times(&anchors, count);
                if anchors.last().is_some_and(|anchor| anchor.1 != count) {
                    warnings.push(format!(
                        "SM {vsmid} of GPU {gpu_index}: anchors end before the last sample."
                    ));
                }
                info.records = count;
                info.anchors = anchors.len();
                info.first_ptimer = anchors.first().map(|anchor| anchor.0);
                info.last_ptimer = anchors.last().map(|anchor| anchor.0);
                for (record, time_ns) in records.as_chunks::<RECORD_SIZE>().0.iter().zip(times) {
                    samples.push(PcSample {
                        gpu_index: gpu_index as u16,
                        vsmid,
                        pc: u64::from_le_bytes(record[..8].try_into().unwrap()),
                        time_ns,
                        warp_slot: record[12],
                        sample_type: record[13],
                        flag: record[14],
                    });
                }
                streams.push(info);
            }
        }
        let profile = ShaderProfile::from_document(document)?;
        if gpus
            .iter()
            .filter(|gpu| gpu.sample_type_table.is_some())
            .count()
            > 1
        {
            warnings.push("Several GPUs have PC samples; shader attribution assumes one shared address space.".into());
        }
        warnings.extend(profile.warnings.iter().cloned());
        Ok(Self {
            gpus,
            streams,
            samples,
            sample_types: sample_types.unwrap_or(&[]),
            profile,
            warnings,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Why the capture has no usable samples, or `None` when it has some.
    pub fn unavailable_reason(&self) -> Option<&'static str> {
        if !self.samples.is_empty() {
            None
        } else if self.streams.is_empty() {
            Some(
                "The capture contains no real-time shader profiler PC samples (gpus[].SamplingData.PcSamplingPerSM is empty); record with --real-time-shader-profiler.",
            )
        } else {
            Some("The capture's PC sample storage could not be decoded; see warnings.")
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn samples(&self) -> &[PcSample] {
        &self.samples
    }

    pub fn profile(&self) -> &ShaderProfile {
        &self.profile
    }

    pub fn sample_types(&self) -> &[SampleType] {
        self.sample_types
    }

    pub fn sample_type_name(&self, code: u8) -> String {
        self.sample_types
            .get(usize::from(code))
            .map_or_else(|| format!("unknown_{code}"), |entry| entry.name.to_owned())
    }

    pub fn summary(&self, scope: &Scope) -> Result<PcSamplingSummary> {
        let selection = self.select(scope)?;
        let total = selection.indices.len();
        let mut overall = GroupCounts::new();
        let mut attribution = AttributionCounts {
            in_code_blocks: 0,
            outside_code_blocks: 0,
            with_spirv_instruction: 0,
            with_source_line: 0,
            with_instruction_class: 0,
        };
        let mut shaders: BTreeMap<ShaderKey, (Vec<usize>, GroupCounts)> = BTreeMap::new();
        for index in &selection.indices {
            let sample = &self.samples[*index];
            let location = self.profile.locate(sample.pc);
            let class = location.and_then(|location| location.correlation.class);
            overall.add(sample.sample_type, class);
            let key = match location {
                Some(location) => {
                    attribution.in_code_blocks += 1;
                    if location.correlation.spirv_index.is_some() {
                        attribution.with_spirv_instruction += 1;
                    }
                    if location.correlation.location.is_some() {
                        attribution.with_source_line += 1;
                    }
                    if class.is_some() {
                        attribution.with_instruction_class += 1;
                    }
                    self.shader_key(location.code_block)
                }
                None => {
                    attribution.outside_code_blocks += 1;
                    ShaderKey::Unattributed
                }
            };
            let entry = shaders
                .entry(key)
                .or_insert_with(|| (Vec::new(), GroupCounts::new()));
            if let Some(location) = location
                && !entry.0.contains(&location.code_block)
            {
                entry.0.push(location.code_block);
            }
            entry.1.add(sample.sample_type, class);
        }
        let mut shader_rows: Vec<_> = shaders
            .into_iter()
            .map(|(key, (mut blocks, counts))| {
                blocks.sort_unstable();
                let samples = counts.total();
                ShaderSamples {
                    shader: self.shader_label(&key),
                    hash: key.hash(),
                    code_blocks: blocks
                        .iter()
                        .map(|block| self.profile.code_blocks[*block].index)
                        .collect(),
                    samples,
                    share_pct: share(samples, total),
                    sample_types: self.type_counts(&counts.types, samples),
                    instruction_mix: class_counts(&counts.classes, samples),
                }
            })
            .collect();
        shader_rows.sort_by_key(|row| std::cmp::Reverse(row.samples));
        let mut warnings = selection.warnings;
        warnings.extend(self.warnings.iter().cloned());
        Ok(PcSamplingSummary {
            available: !self.samples.is_empty(),
            unavailable_reason: self.unavailable_reason(),
            gpus: self.gpus.clone(),
            records: self.samples.len(),
            sm_streams: self
                .streams
                .iter()
                .filter(|stream| stream.records > 0)
                .count(),
            anchors: self.streams.iter().map(|stream| stream.anchors).sum(),
            scope: selection.scope,
            selected_samples: total,
            untimed_samples_excluded: selection.untimed,
            attribution,
            sample_types: self.type_counts(&overall.types, total),
            instruction_mix: class_counts(&overall.classes, total),
            shaders: shader_rows,
            time_attribution: TIME_ATTRIBUTION,
            exact_per_sample_timestamps: false,
            warnings,
        })
    }

    /// Rank source lines, SPIR-V instructions, or SASS instructions by sample count.
    pub fn hotspots(
        &self,
        scope: &Scope,
        group_by: HotspotGrouping,
        shader_filter: Option<&Regex>,
    ) -> Result<HotspotReport> {
        let selection = self.select(scope)?;
        let total = selection.indices.len();
        let mut groups: HashMap<HotspotKey, GroupCounts> = HashMap::new();
        let mut matching = 0usize;
        for index in &selection.indices {
            let sample = &self.samples[*index];
            let location = self.profile.locate(sample.pc);
            let shader = location.map_or(ShaderKey::Unattributed, |location| {
                self.shader_key(location.code_block)
            });
            if let Some(filter) = shader_filter
                && !(filter.is_match(&self.shader_label(&shader))
                    || shader.hash().is_some_and(|hash| filter.is_match(&hash)))
            {
                continue;
            }
            matching += 1;
            let key = match (group_by, location) {
                (_, None) => HotspotKey::Unattributed,
                (HotspotGrouping::Line, Some(location)) => {
                    let module = self.profile.code_blocks[location.code_block].module_index;
                    HotspotKey::Line(shader, module.zip(location.correlation.location))
                }
                (HotspotGrouping::Instruction, Some(location)) => HotspotKey::Instruction(
                    shader,
                    location.code_block,
                    location.correlation.spirv_index,
                ),
                (HotspotGrouping::Sass, Some(location)) => {
                    HotspotKey::Sass(location.code_block, location.offset & !15)
                }
            };
            groups.entry(key).or_insert_with(GroupCounts::new).add(
                sample.sample_type,
                location.and_then(|location| location.correlation.class),
            );
        }
        let mut hotspots: Vec<_> = groups
            .into_iter()
            .map(|(key, counts)| self.hotspot(&key, &counts, total))
            .collect();
        hotspots.sort_by(|left, right| {
            right
                .samples
                .cmp(&left.samples)
                .then_with(|| left.shader.cmp(&right.shader))
                .then_with(|| left.line.cmp(&right.line))
                .then_with(|| left.spirv_index.cmp(&right.spirv_index))
                .then_with(|| left.sass_offset.cmp(&right.sass_offset))
        });
        let mut warnings = selection.warnings;
        warnings.extend(self.warnings.iter().cloned());
        Ok(HotspotReport {
            group_by,
            scope: selection.scope,
            selected_samples: total,
            matching_samples: matching,
            hotspots,
            share_basis: "percent of all samples selected by the scope",
            warnings,
        })
    }

    /// Decode one page of attributed sample records selected by a scope.
    pub fn records(
        &self,
        scope: &Scope,
        vsmid: Option<u32>,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<SampleRecord>, usize, Scope)> {
        let selection = self.select(scope)?;
        let indices: Vec<_> = selection
            .indices
            .into_iter()
            .filter(|index| vsmid.is_none_or(|vsmid| self.samples[*index].vsmid == vsmid))
            .collect();
        let start = offset.min(indices.len());
        let stop = indices.len().min(start.saturating_add(limit));
        let records = indices[start..stop]
            .iter()
            .map(|index| {
                let sample = &self.samples[*index];
                let location = self.profile.locate(sample.pc);
                let block = location.map(|location| &self.profile.code_blocks[location.code_block]);
                let source = location.and_then(|location| {
                    self.source_of(location.code_block, location.correlation.location)
                });
                SampleRecord {
                    index: *index,
                    gpu_index: sample.gpu_index,
                    vsmid: sample.vsmid,
                    pc: sample.pc,
                    time_ns: sample.time_ns,
                    sample_type: sample.sample_type,
                    sample_type_name: self.sample_type_name(sample.sample_type),
                    warp_slot: sample.warp_slot,
                    flag: sample.flag,
                    shader: block.map(|block| self.profile.shader_name(block)),
                    shader_hash: block.and_then(|block| block.module_hash.clone()),
                    code_block: block.map(|block| block.index),
                    sass_offset: location.map(|location| location.offset),
                    instruction_class: location
                        .and_then(|location| location.correlation.class)
                        .map(|class| class.label()),
                    spirv_index: location.and_then(|location| location.correlation.spirv_index),
                    file: source.as_ref().map(|source| file_basename(&source.0)),
                    line: source.map(|source| source.1),
                }
            })
            .collect();
        Ok((records, indices.len(), selection.scope))
    }

    /// Modules and code blocks with their whole-capture sample counts.
    pub fn shaders(&self) -> serde_json::Value {
        let mut per_block = vec![0usize; self.profile.code_blocks.len()];
        let mut outside = 0usize;
        for sample in &self.samples {
            match self.profile.locate(sample.pc) {
                Some(location) => per_block[location.code_block] += 1,
                None => outside += 1,
            }
        }
        let modules: Vec<_> = self
            .profile
            .modules
            .iter()
            .enumerate()
            .map(|(position, module)| {
                let samples: usize = self
                    .profile
                    .code_blocks
                    .iter()
                    .zip(&per_block)
                    .filter(|(block, _)| block.module_index == Some(position))
                    .map(|(_, count)| *count)
                    .sum();
                let mut value = serde_json::to_value(module).unwrap_or_default();
                value["samples"] = samples.into();
                value
            })
            .collect();
        let code_blocks: Vec<_> = self
            .profile
            .code_blocks
            .iter()
            .zip(&per_block)
            .map(|(block, samples)| {
                let mut value = serde_json::to_value(block).unwrap_or_default();
                value["shader"] = self.profile.shader_name(block).into();
                value["samples"] = (*samples).into();
                value
            })
            .collect();
        serde_json::json!({
            "available": !self.samples.is_empty(),
            "unavailable_reason": self.unavailable_reason(),
            "records": self.samples.len(),
            "samples_outside_code_blocks": outside,
            "blob_count": self.profile.blob_count,
            "blob_key_source": self.profile.key_source,
            "modules": modules,
            "code_blocks": code_blocks,
            "spirv_listing_line_rule": format!("spirv_listing_line = spirv_index + {SPIRV_FIRST_INSTRUCTION_LINE}"),
            "warnings": self.warnings,
        })
    }

    fn select(&self, scope: &Scope) -> Result<Selection> {
        let mut warnings = Vec::new();
        if scope.kind == ScopeKind::Capture {
            return Ok(Selection {
                scope: scope.clone(),
                indices: (0..self.samples.len()).collect(),
                untimed: 0,
                warnings,
            });
        }
        if scope.kind == ScopeKind::SampleRange {
            return Err(Error::TracePath(
                "PC samples cannot be selected by counter sample range; use a time, call, frame, or labeled scope".into(),
            ));
        }
        let (start, end) = scope.start_ns.zip(scope.end_ns).ok_or_else(|| {
            Error::TracePath(format!("scope {} has no usable GPU timestamps", scope.id))
        })?;
        if scope.precision == "nvtx_clock_unvalidated" {
            warnings.push("NVTX range clock is not validated against GPU PTIMER; PC sample selection may be wrong.".into());
        }
        let mut untimed = 0usize;
        let indices = self
            .samples
            .iter()
            .enumerate()
            .filter_map(|(index, sample)| match sample.time_ns {
                Some(time) => (start <= time && time < end).then_some(index),
                None => {
                    untimed += 1;
                    None
                }
            })
            .collect();
        if untimed > 0 {
            warnings.push(format!(
                "{untimed} samples without anchor time were excluded from the time scope."
            ));
        }
        Ok(Selection {
            scope: scope.clone(),
            indices,
            untimed,
            warnings,
        })
    }

    fn shader_key(&self, code_block: usize) -> ShaderKey {
        let block = &self.profile.code_blocks[code_block];
        match block.module_hash_value.or_else(|| {
            self.profile
                .module_of(block)
                .and_then(|module| module.hash_value)
        }) {
            Some(hash) => ShaderKey::Module(hash),
            None => ShaderKey::CodeBlock(code_block),
        }
    }

    fn shader_label(&self, key: &ShaderKey) -> String {
        match key {
            ShaderKey::Module(hash) => self
                .profile
                .modules
                .iter()
                .find(|module| module.hash_value == Some(*hash))
                .and_then(|module| module.name.clone())
                .unwrap_or_else(|| format_hash(*hash)),
            ShaderKey::CodeBlock(block) => {
                self.profile.shader_name(&self.profile.code_blocks[*block])
            }
            ShaderKey::Unattributed => "(outside known code blocks)".into(),
        }
    }

    fn source_of(&self, code_block: usize, location: Option<u32>) -> Option<(String, u32)> {
        let block = &self.profile.code_blocks[code_block];
        let module = self.profile.module_of(block)?.spirv.as_ref()?;
        let location = module.locations.get(location? as usize)?;
        Some((location.file.clone(), location.line))
    }

    fn type_counts(&self, counts: &[usize; 256], total: usize) -> Vec<SampleTypeCount> {
        let mut result: Vec<_> = counts
            .iter()
            .enumerate()
            .filter(|(_, samples)| **samples > 0)
            .map(|(code, samples)| {
                let code = code as u8;
                let entry = self.sample_types.get(usize::from(code));
                SampleTypeCount {
                    code,
                    name: entry
                        .map_or_else(|| format!("unknown_{code}"), |entry| entry.name.into()),
                    label: entry
                        .map_or_else(|| format!("Unknown ({code})"), |entry| entry.label.into()),
                    kind: entry.map_or(SampleKind::Unknown, |entry| entry.kind),
                    samples: *samples,
                    share_pct: share(*samples, total),
                }
            })
            .collect();
        result.sort_by(|left, right| {
            right
                .samples
                .cmp(&left.samples)
                .then(left.code.cmp(&right.code))
        });
        result
    }

    fn hotspot(&self, key: &HotspotKey, counts: &GroupCounts, total: usize) -> Hotspot {
        let samples = counts.total();
        let mut hotspot = Hotspot {
            shader: String::new(),
            shader_hash: None,
            file: None,
            path: None,
            line: None,
            spirv_index: None,
            spirv_listing_line: None,
            spirv_instruction: None,
            code_block: None,
            sass_offset: None,
            sass_encoding: None,
            samples,
            share_pct: share(samples, total),
            sample_types: self.type_counts(&counts.types, samples),
            instruction_mix: class_counts(&counts.classes, samples),
        };
        match key {
            HotspotKey::Unattributed => {
                hotspot.shader = self.shader_label(&ShaderKey::Unattributed);
            }
            HotspotKey::Line(shader, location) => {
                hotspot.shader = self.shader_label(shader);
                hotspot.shader_hash = shader.hash();
                if let Some((module, location)) = location
                    && let Some(source) = self.profile.modules[*module]
                        .spirv
                        .as_ref()
                        .and_then(|spirv| spirv.locations.get(*location as usize))
                {
                    hotspot.file = Some(file_basename(&source.file));
                    hotspot.path = Some(source.file.clone());
                    hotspot.line = Some(source.line);
                }
            }
            HotspotKey::Instruction(shader, block, spirv_index) => {
                hotspot.shader = self.shader_label(shader);
                hotspot.shader_hash = shader.hash();
                hotspot.spirv_index = *spirv_index;
                hotspot.spirv_listing_line =
                    spirv_index.map(|index| index + SPIRV_FIRST_INSTRUCTION_LINE);
                let block = &self.profile.code_blocks[*block];
                if let (Some(index), Some(spirv)) = (
                    spirv_index,
                    self.profile
                        .module_of(block)
                        .and_then(|module| module.spirv.as_ref()),
                ) {
                    hotspot.spirv_instruction = spirv.describe(*index as usize);
                    if let Some(source) = spirv
                        .instruction_locations
                        .get(*index as usize)
                        .copied()
                        .flatten()
                        .and_then(|location| spirv.locations.get(location as usize))
                    {
                        hotspot.file = Some(file_basename(&source.file));
                        hotspot.path = Some(source.file.clone());
                        hotspot.line = Some(source.line);
                    }
                }
            }
            HotspotKey::Sass(block_index, offset) => {
                let block = &self.profile.code_blocks[*block_index];
                let shader = self.shader_key(*block_index);
                hotspot.shader = self.shader_label(&shader);
                hotspot.shader_hash = shader.hash();
                hotspot.code_block = Some(block.index);
                hotspot.sass_offset = Some(*offset);
                hotspot.sass_encoding = sass_encoding(block, *offset);
                let correlation = block
                    .instructions
                    .get((*offset / 16) as usize)
                    .copied()
                    .unwrap_or_default();
                hotspot.spirv_index = correlation.spirv_index;
                hotspot.spirv_listing_line = correlation
                    .spirv_index
                    .map(|index| index + SPIRV_FIRST_INSTRUCTION_LINE);
                if let Some(spirv) = self
                    .profile
                    .module_of(block)
                    .and_then(|module| module.spirv.as_ref())
                {
                    hotspot.spirv_instruction = correlation
                        .spirv_index
                        .and_then(|index| spirv.describe(index as usize));
                }
                if let Some(source) = self.source_of(*block_index, correlation.location) {
                    hotspot.file = Some(file_basename(&source.0));
                    hotspot.path = Some(source.0);
                    hotspot.line = Some(source.1);
                }
            }
        }
        hotspot
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ShaderKey {
    Module(u64),
    CodeBlock(usize),
    Unattributed,
}

impl ShaderKey {
    fn hash(&self) -> Option<String> {
        match self {
            Self::Module(hash) => Some(format_hash(*hash)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HotspotKey {
    /// Shader and (module index, location index) of the source line.
    Line(ShaderKey, Option<(usize, u32)>),
    Instruction(ShaderKey, usize, Option<u32>),
    Sass(usize, u64),
    Unattributed,
}

/// Raw 128-bit SASS encoding as two little-endian 64-bit words, like `nvdisasm` comments.
fn sass_encoding(block: &CodeBlock, offset: u64) -> Option<String> {
    let start = usize::try_from(offset).ok()?;
    let bytes = block.sass.get(start..start + 16)?;
    let low = u64::from_le_bytes(bytes[..8].try_into().ok()?);
    let high = u64::from_le_bytes(bytes[8..].try_into().ok()?);
    Some(format!("{low:#018x} {high:#018x}"))
}

/// Instruction mix rows sorted by samples; unclassified samples get `None` fields.
fn class_counts(
    counts: &HashMap<Option<InstructionClass>, usize>,
    total: usize,
) -> Vec<InstructionClassCount> {
    let mut result: Vec<_> = counts
        .iter()
        .map(|(class, samples)| InstructionClassCount {
            label: class.map_or_else(|| "(unclassified)".into(), |class| class.label()),
            pipe: class.map(|class| class.pipe),
            family: class.map(|class| class.family),
            operation: class.and_then(|class| class.operation),
            samples: *samples,
            share_pct: share(*samples, total),
        })
        .collect();
    result.sort_by(|left, right| {
        right
            .samples
            .cmp(&left.samples)
            .then_with(|| left.label.cmp(&right.label))
    });
    result
}

fn share(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * part as f64 / total as f64
    }
}

/// Decode `(PTIMER, cumulative sample index)` anchors and validate their order.
fn decode_anchors(data: &[u8], count: usize) -> Result<Vec<(u64, usize)>> {
    if !data.len().is_multiple_of(ANCHOR_SIZE) {
        return Err(Error::InvalidCapture(format!(
            "timestamp storage has {} bytes, not a multiple of {ANCHOR_SIZE}",
            data.len()
        )));
    }
    let mut anchors = Vec::with_capacity(data.len() / ANCHOR_SIZE);
    let mut previous = (0u64, 0usize);
    for anchor in data.as_chunks::<ANCHOR_SIZE>().0 {
        let ptimer = u64::from_le_bytes(anchor[..8].try_into().unwrap());
        let end = u64::from_le_bytes(anchor[8..].try_into().unwrap());
        let end = usize::try_from(end)
            .ok()
            .filter(|end| *end >= previous.1 && *end <= count)
            .ok_or_else(|| {
                Error::InvalidCapture(
                    "timestamp anchors have an invalid cumulative sample index".into(),
                )
            })?;
        if ptimer < previous.0 {
            return Err(Error::InvalidCapture(
                "timestamp anchors are not monotonic".into(),
            ));
        }
        anchors.push((ptimer, end));
        previous = (ptimer, end);
    }
    Ok(anchors)
}

/// Midpoint time of every sample slot between the anchors that enclose it.
/// Samples before the first anchor get its time; samples after the last none.
fn sample_times(anchors: &[(u64, usize)], count: usize) -> Vec<Option<u64>> {
    let mut times = vec![None; count];
    let mut previous: Option<(u64, usize)> = None;
    for &(ptimer, end) in anchors {
        let start = previous.map_or(0, |anchor| anchor.1);
        let span = (end - start) as u64;
        for (ordinal, time) in times[start..end].iter_mut().enumerate() {
            *time = Some(match previous {
                Some((previous_ptimer, _)) if ptimer > previous_ptimer => {
                    previous_ptimer
                        + ((ptimer - previous_ptimer) as u128 * (2 * ordinal as u128 + 1)
                            / (2 * span as u128)) as u64
                }
                _ => ptimer,
            });
        }
        previous = Some((ptimer, end));
    }
    times
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ada_table_matches_the_exported_metric_order() {
        // GPUTRACE_FRAME exports tpc__warps_issue_stalled_* rows in code order 1..20.
        let exported = [
            "drain",
            "imc_miss",
            "barrier",
            "membar",
            "sleeping",
            "branch_resolving",
            "wait",
            "long_scoreboard_pipe_rtcore",
            "long_scoreboard_pipe_l1tex",
            "short_scoreboard",
            "no_instruction",
            "math_pipe_throttle",
            "tex_throttle",
            "mio_throttle_pipe_rtcore",
            "lg_throttle",
            "mio_throttle_pipe_mio",
            "dispatch_stall",
            "selected",
            "not_selected",
            "misc",
        ];
        let (name, table) = sample_type_table(0x190).unwrap();
        assert_eq!(name, "ada");
        for (code, expected) in exported.iter().enumerate() {
            assert_eq!(table[code + 1].name, *expected);
            assert_eq!(usize::from(table[code + 1].code), code + 1);
        }
        assert!(sample_type_table(0x160).is_none());
        assert_eq!(sample_type_table(0x1b0).unwrap().1.len(), 20);
    }

    #[test]
    fn anchors_interpolate_midpoints_and_reject_disorder() {
        let mut data = Vec::new();
        for (ptimer, end) in [(1000u64, 2u64), (1100, 4), (1100, 5)] {
            data.extend(ptimer.to_le_bytes());
            data.extend(end.to_le_bytes());
        }
        let anchors = decode_anchors(&data, 6).unwrap();
        let times = sample_times(&anchors, 6);
        assert_eq!(
            times,
            [
                Some(1000),
                Some(1000),
                Some(1025),
                Some(1075),
                Some(1100),
                None
            ]
        );

        let mut bad = Vec::new();
        for (ptimer, end) in [(1000u64, 3u64), (900, 4)] {
            bad.extend(ptimer.to_le_bytes());
            bad.extend(end.to_le_bytes());
        }
        assert!(decode_anchors(&bad, 4).is_err());
        assert!(decode_anchors(&data, 4).is_err());
    }
}
